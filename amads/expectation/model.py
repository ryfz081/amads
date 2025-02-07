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
    def __init__(self, order: int, smoothing_factor: float = 0.1):
        self.order = order
        self.smoothing_factor = smoothing_factor
        self.vocabulary = set()
        self.ngrams = {}

    def train(self, corpus) -> None:
        """Train the model by counting N-gram frequencies in the corpus."""
        self.vocabulary = set()
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
                self.vocabulary.add(target)

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
        # Add current_token to vocabulary if it's new
        self.vocabulary.add(current_token)
        V = len(self.vocabulary)  # vocabulary size
        
        if context not in self.ngrams:
            # For unseen contexts, use maximum smoothing (equivalent to uniform)
            # Every token gets the same smoothed probability
            return ProbabilityDistribution({t: 1/V for t in self.vocabulary})
            
        # Get counts and apply smoothing
        probabilities = {}
        total_count = sum(self.ngrams[context].values()) + (self.smoothing_factor * V)
        
        # Calculate smoothed probability for all possible tokens
        for token in self.vocabulary:
            count = self.ngrams[context].get(token, 0)
            probabilities[token] = (count + self.smoothing_factor) / total_count
        
        return ProbabilityDistribution(probabilities)



