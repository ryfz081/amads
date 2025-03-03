import os
import sys
import pickle

# Set up paths for testing -- this is hacky and should be fixed if/when package is properly installed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pytest
import numpy as np
from amads.expectation.model import MarkovEnsemble
from amads.expectation.predictions import Prediction

def run_tests():
    print("\n=== Starting MarkovEnsemble Tests ===\n")

    # # Test initialization
    # print("1. Testing initialization...")
    # model = MarkovEnsemble(min_order=1, max_order=3)
    # assert model.min_order == 1, "min_order not set correctly"
    # assert model.max_order == 3, "max_order not set correctly"
    # print(f"✓ Model initialized with orders from {model.min_order} to {model.max_order}")
    # print(f"✓ Number of models created: {len(model.models)}")
    
    # # Test training
    # print("\n2. Testing training...")
    # model = MarkovEnsemble(min_order=1, max_order=2)
    # sequences = [[1, 2, 1, 2], [1, 2, 1, 3]]
    # print(f"Training sequences: {sequences}")
    # model.train(sequences)
    # assert len(model.models) == 2, "Wrong number of models created"
    # print(f"✓ Successfully trained {len(model.models)} models")
    # print("✓ Model vocabularies:")
    # for i, m in enumerate(model.models, start=1):
    #     print(f"  - Order {i} model vocabulary: {sorted(list(m.vocabulary))}")
    
    # # Test simple prediction
    # print("\n3. Testing simple prediction...")
    # model = MarkovEnsemble(min_order=1, max_order=4, smoothing_factor=0.01)
    # training_seq = [[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]]
    # print(f"Training sequence: {training_seq}")
    # model.train(training_seq)
    
    # context = [1]
    # print(f"Testing prediction with context: {context}")
    # pred = model.predict_token(context)
    # print(f"✓ Prediction distribution: {dict(pred.prediction.distribution)}")
    # assert isinstance(pred, Prediction), "Prediction not returned correctly"
    # assert pred.prediction.distribution[2] > 0.9, "Failed to predict obvious pattern"
    # print(f"✓ Successfully predicted '2' with probability > 0.9")
    
    # Test combination strategies
    print("\n4. Testing combination strategies...")
    # Create a sequence where different orders will definitely disagree
    training_seq = [[1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2, 5, 2, 1, 4, 2, 3, 5, 2, 3, 4]]
    test_context = [4, 1, 2]
    
    print(f"Training sequence: {training_seq}")
    print(f"Test context: {test_context}")
    print("\nComparing predictions across different strategies:")
    
    strategies = ['ppm-a', 'ppm-b', 'ppm-c', 'entropy']
    predictions = {}
    
    for strategy in strategies:
        model = MarkovEnsemble(
            min_order=1, 
            max_order=4, 
            combination_strategy=strategy,
            smoothing_factor=0.01
        )
        model.train(training_seq)
        pred = model.predict_token(test_context)
        predictions[strategy] = dict(pred.prediction.distribution)
        print(f"\n{strategy.upper()} Strategy:")
        print(f"  Prediction: {predictions[strategy]}")
        
        # Print additional debug info
        print(f"  Model details:")
        for order, m in enumerate(model.models, start=1):
            context = tuple(test_context[-order:]) if order <= len(test_context) else tuple()
            if context in m.ngrams:
                print(f"    Order-{order} counts for context {context}: {m.ngrams[context]}")
            else:
                print(f"    Order-{order} has no counts for context {context}")
    
    # Verify that strategies give different results
    assert len(set(str(p) for p in predictions.values())) > 1, "All strategies gave identical predictions"
    print("\n✓ Different strategies produced different predictions as expected")
    
    # Test sequence prediction
    print("\n5. Testing sequence prediction...")
    model = MarkovEnsemble(min_order=1, max_order=2)
    training_seq = [[1, 2, 3, 1, 2, 3]]
    print(f"Training sequence: {training_seq}")
    model.train(training_seq)
    
    test_seq = [1, 2, 3, 1]
    print(f"Test sequence: {test_seq}")
    predictions = model.predict_sequence(test_seq)
    print("\nPredictions for each position:")
    for i, pred in enumerate(predictions.predictions):
        context = test_seq[i:i+model.min_order]
        next_token = test_seq[i+model.min_order]
        print(f"  Context {context} → Predicted: {dict(pred.prediction.distribution)}")
        print(f"  Actual next token was: {next_token}")
    
    assert len(predictions.predictions) == len(test_seq) - 1, "Wrong number of predictions"
    print(f"✓ Generated {len(predictions.predictions)} predictions as expected")
    
    print("\n=== All tests passed successfully! ===")

if __name__ == "__main__":
    try:
        run_tests()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}") 