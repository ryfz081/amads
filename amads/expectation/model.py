from typing import List

from amads.expectation.probability import ProbabilityDistribution
from amads.expectation.tokenizer import Token


class ExpectationModel:
    def train(self, corpus) -> None:
        """
        Train the model on a corpus of music.

        A corpus is a collection of musical sequences.
        """
        raise NotImplementedError

    def predict_sequence(self, sequence) -> None:
        """
        Generate predictions for a musical sequence (e.g. a melody).


        The model predicts one musical sequence at a time.
        """
        # Probably calls predict_token many times?
        raise NotImplementedError

    def predict_token(self, token):
        raise NotImplementedError

class MarkovModel(ExpectationModel):
    def __init__(self, order: int):
        self.order = order

    def train(self, corpus) -> None:
        """Train the model by counting N-gram frequencies in the corpus."""
        self.ngrams = {}
        for sequence in corpus:
            # Convert sequence to tuple for easier handling
            seq = tuple(sequence)
            # For each possible n-gram in the sequence
            for i in range(len(seq) - self.order):
                # Get the context (previous tokens) and the target token
                context = seq[i:i + self.order]
                target = seq[i + self.order]
                
                # Initialize context in ngrams if not present
                if context not in self.ngrams:
                    self.ngrams[context] = {}
                
                # Count the frequency of the target given this context
                self.ngrams[context][target] = self.ngrams[context].get(target, 0) + 1

    def predict_sequence(self, sequence: List[Token]) -> List[ProbabilityDistribution]:
        """Generate predictions for each token in the sequence given previous tokens."""
        predictions = []
        seq = tuple(sequence)
        
        # For each position where we can make a prediction
        for i in range(len(seq) - self.order):
            context = seq[i:i + self.order]
            target = seq[i + self.order]
            pred = self.predict_token(context, target)
            predictions.append(pred)
            
        return predictions

    def predict_token(self, context: tuple, current_token: Token) -> ProbabilityDistribution:
        """Predict probability distribution for next token given context."""
        if context not in self.ngrams:
            # Create uniform distribution over all unique tokens seen in training
            unique_tokens = {token for ngram in self.ngrams.values() for token in ngram}
            return ProbabilityDistribution({t: 1/len(unique_tokens) for t in unique_tokens | {current_token}})
            
        # Get all counts for this context
        total_count = sum(self.ngrams[context].values())
        
        # Calculate probability for all possible next tokens
        probabilities = {}
        for token, count in self.ngrams[context].items():
            probabilities[token] = count / total_count
            
        # If the current_token wasn't seen in this context during training,
        # its probability should be 0
        if current_token not in probabilities:
            probabilities[current_token] = 0.0
        
        return ProbabilityDistribution(probabilities)



