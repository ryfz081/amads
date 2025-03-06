from typing import List, Sequence, Optional
from amads.expectation.models.base_model import ExpectationModel, EnsembleModel
from amads.expectation.probability import ProbabilityDistribution
from amads.expectation.predictions import Prediction, SequencePrediction
from amads.expectation.tokenizer import Token

class MarkovModel(ExpectationModel):
    def __init__(self, order: int, smoothing_factor: float = 0.1):
        self.order = order
        self.smoothing_factor = smoothing_factor
        self.reset()

    def reset(self) -> None:
        """Reset the model to its initial untrained state."""
        self.vocabulary = set()
        self.ngrams = {}

    def update(self, corpus) -> None:
        """Update the model with new data without resetting existing knowledge."""
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
                self.vocabulary.add(target)

    def fit(self, corpus) -> None:
        """Fit the model by counting N-gram frequencies in the corpus. 
        This will reset the model to its initial untrained state before training."""
        self.reset()
        
        # First pass: establish complete vocabulary from corpus
        for sequence in corpus:
            self.vocabulary.update(sequence)
            
        # Second pass: count n-grams
        self.update(corpus)

    def predict_sequence(self, sequence: List[Token]) -> SequencePrediction:
        """Generate predictions for each token in the sequence given previous tokens."""
        predictions = []
        seq = tuple(sequence)
        
        # For each position where we can make a prediction
        for i in range(len(seq) - self.order):
            context = seq[i:i + self.order]
            target = seq[i + self.order]
            predictions.append(self.predict_token(context, target))
            
        return SequencePrediction(predictions)

    def predict_token(self, context: tuple, current_token: Optional[Token] = None) -> Prediction:
        """Predict probability distribution for next token given context."""
        V = len(self.vocabulary)  # vocabulary size
        
        if context not in self.ngrams:
            # For unseen contexts, use maximum smoothing (equivalent to uniform)
            prediction = ProbabilityDistribution({t: 1/V for t in self.vocabulary})
            return Prediction(prediction, observation=current_token)
            
        # Get counts and apply smoothing
        probabilities = {}
        total_count = sum(self.ngrams[context].values()) + (self.smoothing_factor * V)
        
        # Calculate smoothed probability for all possible tokens
        for token in self.vocabulary:
            count = self.ngrams[context].get(token, 0)
            probabilities[token] = (count + self.smoothing_factor) / total_count
        
        prediction = ProbabilityDistribution(probabilities)
        return Prediction(prediction, observation=current_token)

    def update_online(self, context: tuple, target: Token) -> None:
        """Update the model with a single new observation."""
        # For STM models, we should accept new tokens and add them to vocabulary
        # This is different from corpus training, where vocabulary should be fixed
        self.vocabulary.add(target)
        for token in context:
            self.vocabulary.add(token)

        # Initialize context in ngrams if not present
        if context not in self.ngrams:
            self.ngrams[context] = {}
        
        # Count the frequency of the target given this context
        self.ngrams[context][target] = self.ngrams[context].get(target, 0) + 1


class MarkovEnsemble(EnsembleModel):
    """Ensemble of Markov models of different orders that combines their predictions."""
    
    def __init__(self, min_order: int, max_order: int, 
                 combination_strategy: str = 'ppm-a',
                 smoothing_factor: float = 0.1):
        """Initialize a Markov model ensemble.
        
        Args:
            min_order: Minimum order of Markov models
            max_order: Maximum order of Markov models
            combination_strategy: How to combine predictions:
                - 'ppm-a': PPM-A escape method
                - 'ppm-b': PPM-B escape method
                - 'ppm-c': PPM-C escape method
                - 'entropy': Entropy-weighted combination
            smoothing_factor: Smoothing factor for all Markov models
        """
        if combination_strategy not in ['ppm-a', 'ppm-b', 'ppm-c', 'entropy']:
            raise ValueError("combination_strategy must be 'ppm-a', 'ppm-b', 'ppm-c', or 'entropy'")
            
        models = [
            MarkovModel(order=i, smoothing_factor=smoothing_factor)
            for i in range(min_order, max_order + 1)
        ]
        super().__init__(models)

        self.combination_strategy = combination_strategy
        self.min_order = min_order
        self.max_order = max_order
        self.smoothing_factor = smoothing_factor

    def fit(self, corpus) -> None:
        """Fit all models in the ensemble on the corpus."""
        for model in self.models:
            model.fit(corpus)

    def update_online(self, context: Sequence[Token], target: Token) -> None:
        """Update all models with new sequence data.
        
        Args:
            context: The sequence of tokens providing context
            target: The token to predict
        """
        for model in self.models:
            if len(context) >= model.order:  # Only update if we have enough context
                model_context = tuple(context[-model.order:])  # Get appropriate context length
                model.update_online(model_context, target)
                
    def predict_sequence(self, sequence: List[Token]) -> SequencePrediction:
        """Generate predictions for each token in the sequence."""
        predictions = []
        seq = tuple(sequence)
        
        # For each position where we can make a prediction
        for i in range(len(seq) - 1):  # -1 since we need at least one token to predict
            # Use all available context up to current position
            context = seq[:i + 1]  # Context is everything up to current position
            target = seq[i + 1]    # Target is next token
            predictions.append(self.predict_token(context, target))
            
        return SequencePrediction(predictions)

    def predict_token(self, context: Sequence[Token], target: Optional[Token] = None) -> Prediction:
        """Generate predictions using all models and combine them using the specified strategy."""
        predictions = []
        contexts = []
        
        context_len = len(context)
        for i, model in enumerate(self.models):
            order = i + self.min_order
            if context_len >= order:
                order_context = tuple(context[-order:])
                pred = model.predict_token(order_context, target)
                predictions.append(pred.prediction)
                contexts.append(order_context)
        
        combined = self.combine_predictions(predictions, contexts)
        return Prediction(combined, observation=target)

    def combine_predictions(self, predictions: List[ProbabilityDistribution],
                          contexts: List[tuple]) -> ProbabilityDistribution:
        """Combine predictions using the specified strategy."""
        if not predictions:
            return ProbabilityDistribution({})

        if self.combination_strategy == 'entropy':
            return self._entropy_weighted_combine(predictions)
        else:  # PPM variants
            return self._ppm_combine(predictions, contexts)

    def _ppm_combine(self, predictions: List[ProbabilityDistribution],
                    contexts: List[tuple]) -> ProbabilityDistribution:
        """Combine predictions using PPM weighting scheme."""
        # Calculate weights based on context frequencies
        weights = []
        
        for i, context in enumerate(contexts):
            if i < len(self.models):
                model = self.models[i]
                context_counts = model.ngrams.get(context, {})
                
                # Calculate escape probability based on PPM variant
                if self.combination_strategy == 'ppm-a':
                    Tn = sum(context_counts.values())
                    escape_prob = 1 / (1 + Tn) if Tn > 0 else 1.0
                elif self.combination_strategy == 'ppm-b':
                    n = sum(context_counts.values())
                    q = sum(1 for count in context_counts.values() if count == 1)
                    escape_prob = q / n if n > 0 else 1.0
                elif self.combination_strategy == 'ppm-c':
                    n = sum(context_counts.values())
                    t = len(context_counts)
                    escape_prob = t / (n + t) if (n + t) > 0 else 1.0
                
                weight = 1 - escape_prob
                weights.append(weight)

        return self._weighted_combine(predictions, weights)

    def _entropy_weighted_combine(self, predictions: List[ProbabilityDistribution]) -> ProbabilityDistribution:
        """Combine predictions using entropy-based weighting."""
        # Calculate entropy for each distribution
        entropies = [pred.entropy() for pred in predictions]
        
        # Convert entropies to weights (lower entropy -> higher weight)
        total_entropy = sum(entropies)
        if total_entropy == 0:
            weights = [1/len(predictions)] * len(predictions)
        else:
            weights = [1 - (e/total_entropy) for e in entropies]

        return self._weighted_combine(predictions, weights)

    def _weighted_combine(self, predictions: List[ProbabilityDistribution],
                         weights: List[float]) -> ProbabilityDistribution:
        """Helper function to combine distributions using provided weights."""
        # Normalize weights
        if sum(weights) > 0:
            weights = [w/sum(weights) for w in weights]
        else:
            weights = [1/len(predictions)] * len(predictions)
            
        # Combine distributions using weights
        combined = {}
        all_tokens = set()
        for pred in predictions:
            all_tokens.update(pred.distribution.keys())
            
        for token in all_tokens:
            weighted_sum = sum(pred.distribution.get(token, 0.0) * weight 
                             for pred, weight in zip(predictions, weights))
            combined[token] = weighted_sum
                
        return ProbabilityDistribution(combined)
