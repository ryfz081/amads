from typing import List, Sequence, Optional
import math

from amads.expectation.probability import ProbabilityDistribution
from amads.expectation.tokenizer import Token
from amads.expectation.predictions import Prediction, SequencePrediction


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
        raise NotImplementedError

    def predict_token(self, token):
        raise NotImplementedError

class EnsembleModel(ExpectationModel):
    """Base class for models that combine multiple sub-models.
    
    This class provides a framework for combining predictions from multiple models,
    without making assumptions about the interface or requirements of those models.
    """
    
    def __init__(self, models):
        """Initialize ensemble with a collection of models.
        
        Args:
            models: Collection of models to be ensembled
        """
        self.models = models

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

    def train(self, corpus) -> None:
        """Train the model by counting N-gram frequencies in the corpus. 
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

    def train(self, corpus) -> None:
        """Train all models in the ensemble on the corpus."""
        for model in self.models:
            model.train(corpus)

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

    def train(self, corpus) -> None:
        """Train the LTM component on the corpus and reset the STM."""
        # Train LTM on full corpus
        self.ltm.train(corpus)
        
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
        
        # Combine distributions
        combined = {}
        all_tokens = set(ltm_dist.distribution.keys()) | set(stm_dist.distribution.keys())
        
        for token in all_tokens:
            ltm_prob = ltm_dist.distribution.get(token, 0.0)
            stm_prob = stm_dist.distribution.get(token, 0.0)
            combined[token] = (ltm_prob * weights[0] + stm_prob * weights[1])
            
        return ProbabilityDistribution(combined)
    



# class IDyOMModel(ExpectationModel):
#     def __init__(self, max_order: int = 8, smoothing_factor: float = 0.1, bias: float = 1.0):
#         """
#         Initialize IDyOM model with both long-term and short-term components.
        
#         Args:
#             max_order: Maximum order of Markov models to use
#             smoothing_factor: Smoothing factor for probability distributions
#             bias: Exponential bias parameter for entropy-based weighting.
#                  - bias > 1: Sharpens the weighting, giving even more influence to the more
#                    confident model (lower entropy)
#                  - bias < 1: Smooths the weighting, making the models' influences more equal
#                    regardless of their confidence
#                  - bias = 1: Direct use of inverse normalized entropy (default)
#         """
#         self.max_order = max_order
#         self.smoothing_factor = smoothing_factor
#         self.bias = bias
        
#         # Long-term models (trained on full corpus)
#         self.ltm_models = [
#             MarkovModel(order=i, smoothing_factor=smoothing_factor)
#             for i in range(1, max_order + 1)
#         ]
        
#         # Short-term models (trained incrementally)
#         self.stm_models = [
#             MarkovModel(order=i, smoothing_factor=smoothing_factor)
#             for i in range(1, max_order + 1)
#         ]

#     def train(self, corpus) -> None:
#         """Train only the Long-Term Models (LTM) on the corpus."""
#         # Train LTM models first (and build vocabulary)
#         for model in self.ltm_models:
#             model.train(corpus)
        
#         # Share the vocabulary with STM models
#         full_vocab = set()
#         for model in self.ltm_models:
#             full_vocab.update(model.vocabulary)
        
#         # Reset STM models but give them the full vocabulary
#         for model in self.stm_models:
#             model.reset()
#             model.vocabulary = full_vocab.copy()  # Give each STM model the full vocabulary

#     def predict_token(self, context: Sequence[Token], target: Token = None) -> Prediction:
#         """Predict the next token given a context using both LTM and STM models."""
#         # Get predictions from all available orders of LTM and STM
#         ltm_predictions = []
#         stm_predictions = []
        
#         context_len = len(context)
#         for order in range(1, min(context_len + 1, self.max_order + 1)):
#             model_idx = order - 1
#             order_context = context[-order:]
            
#             ltm_pred = self.ltm_models[model_idx].predict_token(order_context, target)
#             stm_pred = self.stm_models[model_idx].predict_token(order_context, target)
            
#             ltm_predictions.append(ltm_pred.prediction)  # Note: accessing the ProbabilityDistribution
#             stm_predictions.append(stm_pred.prediction)
            
#         # Combine predictions using entropy weighting
#         combined_ltm = self.entropy_weighted_combination(ltm_predictions)
#         combined_stm = self.entropy_weighted_combination(stm_predictions)
        
#         # Final combination of LTM and STM
#         final_prediction = self.entropy_weighted_combination([combined_ltm, combined_stm])
#         return Prediction(final_prediction, observation=target)

#     def predict_sequence(self, sequence: List[Token]) -> SequencePrediction:
#         """Generate predictions for each token in the sequence given previous tokens."""
#         predictions = []
#         seq = tuple(sequence)
        
#         # For each position where we can make a prediction (starting after first note)
#         for i in range(len(seq) - 1):
#             # Make prediction first
#             context = seq[:i + 1]  # Use all available context
#             target = seq[i + 1]
#             predictions.append(self.predict_token(context, target))
            
#             # Then update STM models with the new observation
#             for model in self.stm_models:
#                 if len(context) >= model.order:
#                     model_context = context[-(model.order):]
#                     model.update_online(model_context, target)
            
#         return SequencePrediction(predictions)
    
#     def entropy_weighted_combination(self, predictions: List[ProbabilityDistribution]) -> ProbabilityDistribution:
#         """Combine predictions using entropy-based weighting."""
#         if not predictions:
#             return ProbabilityDistribution({})
            
#         # Calculate entropy for each distribution
#         entropies = [pred.entropy() for pred in predictions]
            
#         # Convert entropies to weights (lower entropy -> higher weight)
#         total_entropy = sum(entropies)
#         if total_entropy == 0:
#             weights = [1/len(predictions)] * len(predictions)
#         else:
#             weights = [1 - (e/total_entropy) for e in entropies]
#             weight_sum = sum(weights)
#             if weight_sum > 0:
#                 weights = [w/weight_sum for w in weights]
#             else:
#                 weights = [1/len(predictions)] * len(predictions)
        
#         # Combine distributions using weights
#         combined = {}
#         all_tokens = set()
#         for pred in predictions:
#             all_tokens.update(pred.distribution.keys())
            
#         for token in all_tokens:
#             weighted_sum = sum(pred.distribution.get(token, 0.0) * weight 
#                              for pred, weight in zip(predictions, weights))
#             combined[token] = weighted_sum
                
#         return ProbabilityDistribution(combined)


#     def ppm_a_combination(self, predictions: List[ProbabilityDistribution], 
#                          contexts: List[tuple]) -> ProbabilityDistribution:
#         """Combine predictions using PPM-A weighting scheme.
        
#         The weight for each order n is:
#         an = 1 - 1/(1 + Tn)
        
#         where Tn is the number of times the context has been seen.
#         As Tn → ∞, an → 1 (rely on higher order)
#         As Tn → 0, an → 0 (escape to lower order)
        
#         Args:
#             predictions: List of probability distributions from different orders
#             contexts: List of contexts used for each prediction (ei-1)
#         """
#         if not predictions:
#             return ProbabilityDistribution({})
            
#         # Calculate weights based on context frequencies
#         weights = []  # an for each order
#         for i, context in enumerate(contexts):
#             if i < len(self.ltm_models):
#                 model = self.ltm_models[i]
#                 # Tn(ei-1) = Σ counts for all continuations after context
#                 Tn = sum(model.ngrams.get(context, {}).values())
#                 # an = 1 - 1/(1 + Tn)
#                 an = 1 - (1 / (1 + Tn)) if Tn > 0 else 0
#                 weights.append(an)
        
#         # Normalize weights to create proper probability distribution
#         if sum(weights) > 0:
#             weights = [w/sum(weights) for w in weights]
#         else:
#             # If no context seen, use uniform weighting
#             weights = [1/len(predictions)] * len(predictions)
        
#         # Combine distributions using weights
#         combined = {}
#         all_tokens = set()
#         for pred in predictions:
#             all_tokens.update(pred.distribution.keys())
            
#         for token in all_tokens:
#             # P(ei|ei-1) = Σ an * Pn(ei|ei-1)
#             weighted_sum = sum(pred.distribution.get(token, 0.0) * weight 
#                              for pred, weight in zip(predictions, weights))
#             combined[token] = weighted_sum
                
#         return ProbabilityDistribution(combined)
