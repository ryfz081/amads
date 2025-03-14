'''Plots expectation trajectories'''

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from amads.expectation.predictions import SequencePrediction
from amads.expectation.tokenizer import Token
from amads.expectation.metrics import NegativeLogLikelihood, Entropy

def plot_expectation_trajectory(
    predictions: SequencePrediction,
    tokens: List[Token],
    metrics: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 6)
) -> None:
    """Plot expectation metrics over time for a sequence of predictions.
    
    Parameters
    ----------
    predictions : SequencePrediction
        Model predictions for the sequence
    tokens : List[Token]
        The sequence of tokens that were predicted (including timestamps)
    metrics : List[str], optional
        List of metrics to plot. Default is ['nll', 'entropy']
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size (width, height)
    """
    if metrics is None:
        metrics = ['nll', 'entropy']
    
    # Initialize metric calculators
    metric_calculators = {
        'nll': NegativeLogLikelihood(),
        'entropy': Entropy()
    }
    
    # Calculate metrics
    metric_values = {}
    for metric_name in metrics:
        if metric_name not in metric_calculators:
            raise ValueError(f"Unknown metric: {metric_name}")
        calculator = metric_calculators[metric_name]
        values = calculator.compute(predictions, tokens[1:])
        metric_values[metric_name] = values
    
    # Get timestamps for x-axis (skip first token as it's context)
    timestamps = []
    last_known_time = 0  # Default to 0 if no timestamps are available
    for token in tokens[1:]:
        if token.start_time is not None:
            last_known_time = token.start_time
        timestamps.append(last_known_time)
    
    # Create plot
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for ax, metric_name in zip(axes, metrics):
        values = metric_values[metric_name]
        ax.plot(timestamps, values, '-', label=metric_name.upper())
        ax.set_ylabel(metric_name.upper())
        ax.grid(True)
        
        # Calculate and display mean
        mean_value = np.mean(values)
        ax.axhline(y=mean_value, color='r', linestyle='--', alpha=0.5)
        ax.text(0.02, 0.95, f'Mean: {mean_value:.2f}', 
                transform=ax.transAxes, 
                verticalalignment='top')
    
    # Set title and labels
    if title:
        fig.suptitle(title)
    axes[-1].set_xlabel('Time')
    
    plt.tight_layout()
    return fig, axes

