from typing import List, Sequence, Optional

from amads.expectation.models.base_model import EnsembleModel
from amads.expectation.probability import ProbabilityDistribution
from amads.expectation.predictions import Prediction, SequencePrediction
from amads.expectation.tokenizer import Token
from amads.expectation.models.statistical.markov import MarkovEnsemble

class IDyOMModel(EnsembleModel):
    """A simplified IDyOM implementation using MarkovEnsemble for both LTM and STM components."""
    
    def __init__(self, min_order: int = 1, max_order: int = 8, 
                 smoothing_factor: float = 0.01,
                 combination_strategy: str = 'ppm-c'):
        """Initialize IDyOM with both LTM and STM components using MarkovEnsemble.
        
        Args:
            min_order: Minimum order of Markov models
            max_order: Maximum order of Markov models
            smoothing_factor: Smoothing factor for probability distributions
            combination_strategy: Strategy for combining predictions from different order models
                ('entropy', 'ppm-a', 'ppm-b', or 'ppm-c')
        """
        self.min_order = min_order
        self.max_order = max_order
        
        # Create LTM and STM components as MarkovEnsembles
        self.ltm = MarkovEnsemble(
            min_order=min_order,
            max_order=max_order,
            smoothing_factor=smoothing_factor,
            combination_strategy=combination_strategy
        )
        
        self.stm = MarkovEnsemble(
            min_order=min_order,
            max_order=max_order,
            smoothing_factor=smoothing_factor,
            combination_strategy=combination_strategy
        )

    def fit(self, corpus) -> None:
        """Fit the LTM component on the corpus and reset the STM."""
        # Fit LTM on full corpus
        self.ltm.fit(corpus)
        
        # Reset STM but ensure it has the same vocabulary as LTM
        self.stm = MarkovEnsemble(
            min_order=self.min_order,
            max_order=self.max_order,
            smoothing_factor=self.ltm.smoothing_factor,
            combination_strategy=self.ltm.combination_strategy
        )
        
        # Share vocabulary across all STM models
        full_vocab = set()
        for ltm_model in self.ltm.models:
            full_vocab.update(ltm_model.vocabulary)
        
        for stm_model in self.stm.models:
            stm_model.vocabulary = full_vocab.copy()

    def predict_sequence(self, sequence: List[Token]) -> SequencePrediction:
        """Generate predictions for each token in the sequence."""
        predictions = []
        seq = tuple(sequence)
        
        # For each position where we can make a prediction
        for i in range(len(seq) - 1):
            # Make prediction first
            context = seq[:i + 1]
            target = seq[i + 1]
            predictions.append(self.predict_token(context, target))
            
            # Update STM with the actual token that occurred
            if i > 0:  # Only update after we have some context
                self.stm.update_online(context, target)
            
        return SequencePrediction(predictions)

    def predict_token(self, context: Sequence[Token], target: Optional[Token] = None) -> Prediction:
        """Generate predictions using both LTM and STM models."""
        # Get predictions from both models
        ltm_pred = self.ltm.predict_token(context, target)
        stm_pred = self.stm.predict_token(context, target)
        
        # Combine LTM and STM predictions using entropy-weighted combination
        combined = self._combine_ltm_stm(ltm_pred.prediction, stm_pred.prediction)
        return Prediction(combined, observation=target)

    def _combine_ltm_stm(self, ltm_dist: ProbabilityDistribution, 
                        stm_dist: ProbabilityDistribution) -> ProbabilityDistribution:
        """Combine LTM and STM predictions using entropy-weighted combination."""
        # If STM distribution is empty (early in sequence), return LTM
        if not stm_dist.distribution:
            return ltm_dist
            
        # Use entropy-weighted combination
        entropies = [ltm_dist.entropy(), stm_dist.entropy()]
        total_entropy = sum(entropies)
        
        if total_entropy == 0:
            weights = [0.5, 0.5]
        else:
            weights = [1 - (e/total_entropy) for e in entropies]
            weight_sum = sum(weights)
            weights = [w/weight_sum for w in weights]
        
        # Get full vocabulary
        vocab = set().union(*(model.vocabulary for model in self.ltm.models))
        V = len(vocab)
        smoothing = self.ltm.smoothing_factor
        
        # Combine distributions with smoothing
        combined = {}
        for token in vocab:
            # We use 1/V as the default probability for unseen tokens instead of 0.0
            ltm_prob = ltm_dist.distribution.get(token, 1/V)
            stm_prob = stm_dist.distribution.get(token, 1/V)
            combined[token] = (ltm_prob * weights[0] + stm_prob * weights[1])
        
        return ProbabilityDistribution(combined)
    


