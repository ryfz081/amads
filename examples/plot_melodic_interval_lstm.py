"""
Plot the melodic interval expectation of a sequence using an LSTM model.

This example demonstrates how to train an LSTM model to predict melodic intervals
in a sequence and compare its performance to IDyOM and Markov models.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from amads.expectation.model import LSTM
from amads.expectation.tokenizer import MelodyIntervalTokenizer
from amads.expectation.dataset import ScoreDataset
from amads.expectation.metrics import NegativeLogLikelihood
from amads.expectation.model import MarkovModel, IDyOMModel
from amads.utils import check_python_package_installed

check_python_package_installed('torch')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Load the same dataset as in the other example
package_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(package_dir, '..', 'amads', 'music', 'Marion_2024_Bach_Chorales.pkl')
score_data = pickle.load(open(data_path, 'rb'))

# Prepare data
tokenizer = MelodyIntervalTokenizer()
dataset = ScoreDataset(score_data, tokenizer)

# Split into train and test sets
test_sequences = [dataset[-1]]  # Take the last sequence for testing
train_sequences = dataset[:-1]  # Use remaining sequences for training

# Initialize and train models
lstm = LSTM(embedding_dim=16, hidden_dim=32)
idyom = IDyOMModel(max_order=7, smoothing_factor=0.01, combination_strategy='ppm-c')
markov = MarkovModel(order=7, smoothing_factor=0.01)

# Train all models
lstm.fit(train_sequences, n_epochs=300, batch_size=32, validation_size=0.1)
idyom.fit(train_sequences)
markov.fit(train_sequences)

# Optional: Plot training history
lstm.plot_training_history()

# Get test sequence predictions from all models
test_sequence = test_sequences[0]
lstm_predictions = lstm.predict_sequence(test_sequence)
idyom_predictions = idyom.predict_sequence(test_sequence)
markov_predictions = markov.predict_sequence(test_sequence)

# Calculate NLL for each model
nll_metric = NegativeLogLikelihood()
lstm_nll = nll_metric.compute(lstm_predictions, test_sequence[1:])
idyom_nll = nll_metric.compute(idyom_predictions, test_sequence[1:])
markov_nll = nll_metric.compute(markov_predictions, test_sequence[markov.order:])

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

# Plot intervals
intervals = [token.value for token in test_sequence]
ax1.plot(range(len(intervals)), intervals, 'k-', label='Melody intervals')
ax1.set_ylabel('Interval size')
ax1.grid(True)
ax1.legend()

# Plot NLL for all models
ax2.plot(range(1, len(test_sequence)), lstm_nll, 'g-', 
         label='LSTM', alpha=0.7)
ax2.plot(range(1, len(test_sequence)), idyom_nll, 'b-', 
         label='IDyOM', alpha=0.7)
ax2.plot(range(markov.order, len(test_sequence)), markov_nll, 'r-', 
         label='Markov', alpha=0.7)
ax2.set_xlabel('Position in sequence')
ax2.set_ylabel('Negative Log Likelihood')
ax2.grid(True)
ax2.legend()

# Add mean NLL values to title
mean_lstm_nll = sum(lstm_nll) / len(lstm_nll)
mean_markov_nll = sum(markov_nll) / len(markov_nll)
mean_idyom_nll = sum(idyom_nll) / len(idyom_nll)
ax2.set_title(f'Mean NLL - LSTM: {mean_lstm_nll:.2f}, '
              f'Markov: {mean_markov_nll:.2f}, '
              f'IDyOM: {mean_idyom_nll:.2f}')

plt.tight_layout()
plt.show() 