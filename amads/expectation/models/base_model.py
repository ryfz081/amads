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

