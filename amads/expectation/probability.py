from typing import Dict
from math import log2
from amads.expectation.tokenizer import Token

class ProbabilityDistribution:
    def __init__(self, distribution: Dict[Token, float]):
        self.distribution = distribution

    def __getitem__(self, token: Token) -> float:
        return self.distribution[token]

    def __len__(self):
        """Return the number of possible outcomes in the distribution."""
        return len(self.distribution)

    def items(self):
        """Returns a view of the distribution's (token, probability) pairs."""
        return self.distribution.items()

    def entropy(self) -> float:
        """Calculate Shannon entropy E(X) of the probability distribution."""
        entropy = 0
        for prob in self.distribution.values():
            if prob > 0:  # Avoid log(0)
                entropy -= prob * log2(prob)
        return entropy

    def normalized_entropy(self) -> float:
        """Calculate normalized entropy NE(X) = E(X)/Emax(X)."""
        shannon_entropy = self.entropy()
        max_entropy = log2(len(self.distribution)) if self.distribution else 0
        return shannon_entropy / max_entropy if max_entropy > 0 else 0
