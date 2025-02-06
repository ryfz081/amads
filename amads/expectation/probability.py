from typing import Dict
from amads.expectation.tokenizer import Token

class ProbabilityDistribution:
    def __init__(self, distribution: Dict[Token, float]):
        self.distribution = distribution

    def __getitem__(self, token: Token) -> float:
        return self.distribution[token]
