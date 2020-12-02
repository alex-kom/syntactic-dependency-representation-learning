"""
Script for training and evaluating
a dependency lstm classifier on the
TREC Question Classification dataset.
"""

import numpy as np
import random

from dependency_trees import DepFeatureGenerator
from data_preprocessing import DatasetCreator
from dependency_lstm import DependencyLSTM


# Read the dependency trees.
trees = DepFeatureGenerator(fnames='TrecQC/trecqc_dep_trees.txt').get_trees()

# Split the dataset into train, validation and test sets.
indexes = list(range(len(trees)))
test_indexes = set(indexes[-500:])
indexes = indexes[:-500]
random.shuffle(indexes)
subset_indexes = {
        'train': set(indexes[500:]),
        'val': set(indexes[:500]),
        'test': test_indexes,
}

# Load pre-trained embeddings, vocabulary and targets.
vecs = np.load('TrecQC/trecqc_extvec_float16.npy').astype('float32')
vocab = open('TrecQC/trecqc_vocab.txt', encoding='utf8').read().splitlines()
targets = open('TrecQC/trecqc_targets.txt', encoding='utf8').read().splitlines()

# Create dataset artifacts for training.
data = DatasetCreator(trees=trees,
                      targets=targets,
                      embeddings=vecs,
                      vocabulary=vocab,
                      subset_indexes=subset_indexes
                      ).make_data()

# Train and evaluate a dependency lstm model with default parameters.
deplstm = DependencyLSTM()
deplstm.train(data)
test_loss, test_accuracy = deplstm.evaluate()
print(f"Accuracy on test set: {test_accuracy}")
