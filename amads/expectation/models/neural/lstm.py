from typing import List, Sequence, Optional
from amads.expectation.models.base_model import ExpectationModel
from amads.expectation.probability import ProbabilityDistribution
from amads.expectation.predictions import Prediction, SequencePrediction
from amads.expectation.tokenizer import Token
from amads.utils import check_python_package_installed

try:
    check_python_package_installed('torch')
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
except ImportError:
    class LSTM(ExpectationModel):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required to use the LSTM model. "
                "Please install it with: pip install torch"
            )

class LSTM(nn.Module, ExpectationModel):
    """LSTM model for sequence prediction.
    
    This model uses an embedding layer followed by an LSTM layer and a fully
    connected output layer to predict the next token in a sequence.
    
    Parameters
    ----------
    embedding_dim : int, optional
        Dimension of token embeddings
    hidden_dim : int, optional
        Dimension of LSTM hidden state
    device : str, optional
        Device to run the model on ('cpu' or 'cuda')
    """
    def __init__(self, embedding_dim: int = 8, hidden_dim: int = 16, device: str = 'cpu'):
        nn.Module.__init__(self)
        ExpectationModel.__init__(self)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.token_to_id = None
        self.vocab_size = None
        self.model_initialized = False
        
        # Initialize loss tracking
        self.train_losses = []
        self.val_losses = []
    
    def _initialize_model(self, corpus):
        """Initialize the model's vocabulary and neural network layers based on the corpus."""
        # Build vocabulary from all sequences
        all_tokens = []
        for seq in corpus:
            all_tokens.extend(seq)
        unique_values = sorted(set(token.value for token in all_tokens))
        
        # Neural networks need numerical indices - map each token value to a unique index
        # Reserve 0 for padding
        self.token_to_id = {value: idx for idx, value in enumerate(unique_values, start=1)}
        self.id_to_token = {idx: value for value, idx in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id) + 1  # +1 for padding
        
        # Create neural layers that expect indices as input
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).to(self.device)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True).to(self.device)
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size).to(self.device)
        self.model_initialized = True
    
    def _prepare_sequence(self, sequence):
        """Convert a sequence of tokens to tensor of indices."""
        # Convert tokens to indices, using index 1 for any tokens not in vocabulary
        return torch.tensor([self.token_to_id.get(token.value, 1) for token in sequence])
    
    def _collate_fn(self, batch):
        """Simple padding of sequences to max length in batch."""
        # Get max length in this batch
        max_len = max(len(seq) for seq in batch)
        
        # Pad all sequences to max_len
        padded_batch = []
        for seq in batch:
            seq_ids = [self.token_to_id.get(token.value, 1) for token in seq]
            padding = [0] * (max_len - len(seq_ids))
            padded_batch.append(seq_ids + padding)
        
        return torch.tensor(padded_batch)

    def forward(self, x):
        """Simple forward pass."""
        if not self.model_initialized:
            raise RuntimeError("Model must be trained before forward pass")
        
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out)

    def fit(self, corpus, n_epochs: int = 300, batch_size: int = 32,
             validation_size: float = 0.1, patience: int = 10, seed: int = 1234) -> None:
        # Set random seeds for reproducibility
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # for multi-GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if not self.model_initialized:
            self._initialize_model(corpus)
        
        # Split into train and validation sets
        val_size = int(len(corpus) * validation_size)
        train_size = len(corpus) - val_size
        generator = torch.Generator().manual_seed(seed)
        train_data, val_data = random_split(corpus, [train_size, val_size], generator=generator)
        
        # Create data loaders
        train_loader = DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        val_loader = DataLoader(
            val_data, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        # Training setup
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding tokens
        optimizer = torch.optim.Adam(self.parameters())
        
        # Training loop
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(n_epochs):
            # Training phase
            self.train()
            total_train_loss = 0
            
            for sequences in train_loader:
                optimizer.zero_grad()
                
                # Get predictions
                output = self(sequences[:, :-1])
                targets = sequences[:, 1:]
                
                # Compute loss
                loss = criterion(output.reshape(-1, self.vocab_size), targets.reshape(-1))
                
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            # Validation phase
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for sequences in val_loader:
                    output = self(sequences[:, :-1])
                    targets = sequences[:, 1:]
                    loss = criterion(output.reshape(-1, self.vocab_size), targets.reshape(-1))
                    total_val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}:')
                print(f'  Training Loss: {avg_train_loss:.4f}')
                print(f'  Validation Loss: {avg_val_loss:.4f}')
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)

    def predict_sequence(self, sequence: List[Token]) -> SequencePrediction:
        """Generate predictions for each token in the sequence."""
        if not self.model_initialized:
            raise RuntimeError("Model must be trained before making predictions")
        predictions = []
        
        # Convert sequence to indices, using 1 (unknown token) for OOV tokens
        seq = torch.tensor([[self.token_to_id.get(t.value, 1) for t in sequence[:-1]]])
        
        self.eval()
        with torch.no_grad():
            output = self(seq)
            probs = torch.softmax(output[0], dim=-1)
            
            for step_probs, target in zip(probs, sequence[1:]):
                distribution = {}
                
                # Add probabilities for known tokens
                for value, idx in self.token_to_id.items():
                    distribution[Token(value)] = step_probs[idx].item()
                
                # Add probability for unknown tokens (index 1)
                unknown_prob = step_probs[1].item()
                
                # If target is OOV, add it with the unknown probability
                if target.value not in self.token_to_id:
                    distribution[target] = unknown_prob
                
                predictions.append(Prediction(
                    ProbabilityDistribution(distribution), 
                    observation=target))
        
        return SequencePrediction(predictions)

    def predict_token(self, context: Sequence[Token], 
                     current_token: Optional[Token] = None) -> Prediction:
        """Predict probability distribution for next token given context."""
        if not self.model_initialized:
            raise RuntimeError("Model must be trained before making predictions")
        
        # If current token is OOV, use uniform distribution
        if current_token and current_token.value not in self.token_to_id:
            uniform_prob = 1.0 / len(self.token_to_id)
            distribution = {Token(value): uniform_prob for value in self.id_to_token.values()}
            distribution[current_token] = uniform_prob
            return Prediction(ProbabilityDistribution(distribution), 
                            observation=current_token)
        
        # Otherwise use model's predictions
        context_tensor = torch.tensor([[self.token_to_id.get(t.value, 1) for t in context]])
        
        self.eval()
        with torch.no_grad():
            output = self(context_tensor)
            probs = torch.softmax(output[0, -1], dim=-1)
            
            distribution = {}
            for value, idx in self.token_to_id.items():
                distribution[Token(value)] = probs[idx].item()
            
        return Prediction(ProbabilityDistribution(distribution), 
                         observation=current_token)

    def get_training_history(self) -> dict[str, list[float]]:
        """Get the training and validation loss history.
        
        Returns:
            Dictionary containing lists of training and validation losses
        """
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def plot_training_history(self) -> None:
        """Plot the training and validation loss curves.
        
        Requires matplotlib to be installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting training history")
        
        if not self.train_losses or not self.val_losses:
            raise ValueError("No training history available. Train the model first.")
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()