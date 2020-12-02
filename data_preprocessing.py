import numpy as np
import tensorflow as tf
from collections import Counter
from dependency_trees import get_dep_context


class DatasetCreator:
    """
    Create dataset artifacts to build a dependency sentence classifier.

    Arguments:

        trees : array-like of list
            Parsed dependency trees container. Tree objects consist of lists
            of DepNode objects, such as those created by
            DepFeatureGenerator.dep_trees.

        targets : array-like of hashable
            Classification targets for each tree.

        embeddings : ndarray of shape (n_tokens, embedding_dim).
            Matrix of pre-trained word and dependency embeddings.

        vocabulary : array-like of str
            Contains the corresponding token for each row of the embedding
            matrix.

        subset_indexes : dict
            Dict containing sets of indexes for train, validation and test
            subsets.

        make_embeddings_subset : bool
            Whether to include only the subset of pre-trained embeddings found
            in the dataset in the embedding matrix and vocabulary. Useful for
            running an experiment with a smaller embedding matrix when all the
            samples are known in advance.

        oov_vec_samples : int
            The number of embeddings to be used to create an out-of-vocabulary
            embedding.
    """

    def __init__(self,
                 trees,
                 targets,
                 embeddings,
                 vocabulary,
                 subset_indexes,
                 make_embeddings_subset=True,
                 oov_vec_samples=100,
                 max_sent_len=None,
                 max_dep_len=None):
        self.trees = trees
        self.targets = targets
        self.embeddings = embeddings
        self.token2idx = {token: idx for idx, token in enumerate(vocabulary)}
        self.train_indexes = subset_indexes['train']
        self.val_indexes = subset_indexes['val']
        self.test_indexes = subset_indexes['test']
        self.oov_vec_samples = oov_vec_samples
        self.make_embeddings_subset = make_embeddings_subset

    def make_oov_vector(self):
        """
        Make and return an out-of-vocabulary embedding.

        The out-of-vocabulary embedding is computed by taking the mean
        of oov_vec_samples embeddings with the least number of occurrences in
        the training set.
        """
        vecs = []
        for token, count in self.train_word_counts.most_common()[::-1]:
            if token in self.token2idx:
                vecs.append(self.embeddings[self.token2idx[token]])
            if len(vecs) == self.oov_vec_samples:
                break
        return np.mean(vecs, axis=0, dtype=np.float32)

    def get_tree_features(self, tree):
        """Return lists of word and dependency tokens of a dependency tree."""
        words = []
        deps = []
        for node in tree:
            context = get_dep_context(node, tree)
            word = context[0]
            words.append(self.dataset_voc[word]
                         if word in self.dataset_voc
                         else self.dataset_voc['<UNK>'])
            deps.append([self.dataset_voc[dep]
                         for dep in context[1:]
                         if dep in self.dataset_voc])
        return words, deps

    def make_data(self):
        """
        Return a dictionary containing dataset artifacts for classification.

        The dictionary contains training, validation and test arrays,
        the dataset embedding matrix, a dictionary mapping token strings to
        row indexes of the embedding matrix, and a dictionary mapping targets
        to indexes.
        """
        self.process_tokens()
        self.make_embeddings()
        self.index_targets()
        return {
            'target2idx': self.target2idx,
            'train_arrays': self.make_arrays(self.train_indexes),
            'val_arrays': self.make_arrays(self.val_indexes),
            'test_arrays': self.make_arrays(self.test_indexes),
            'embeddings': self.dataset_vecs,
            'vocabulary': self.dataset_voc,
        }

    def index_targets(self):
        """Make a dict mapping targets to indexes."""
        self.target2idx = {
                target: idx for idx, target in enumerate(set(self.targets))
        }

    def process_tokens(self):
        """
        Create a set of the dataset tokens and a Counter of training tokens.
        """
        self.tokens = set()
        self.train_word_counts = Counter()
        for idx, tree in enumerate(self.trees):
            for node in tree:
                dep_context = get_dep_context(node, tree)
                self.tokens.update(dep_context)
                if idx in self.train_indexes:
                    self.train_word_counts.update(dep_context[0])

    def make_arrays(self, sample_indexes):
        """Return input and target arrays for given sample_indexes.

        The returned input consists of a list of two arrays:
            word_inputs with shape (n_samples, max_sent_len) and
            dep_inputs with shape (n_samples, max_sent_len, max_dep_len)
        max_sent_len and max_dep_len will be the maximum length of words
        and dependencies found in the samples.

        The target array has shape (n_samples,) if the number of targets
        is two, or shape (n_samples, num_classes) otherwise.
        """
        word_inputs = []
        dep_inputs = []
        targets = []
        for idx in sorted(sample_indexes):
            words, deps = self.get_tree_features(self.trees[idx])
            word_inputs.append(words)
            dep_inputs.append(deps)
            targets.append(self.target2idx[self.targets[idx]])

        word_inputs = tf.keras.preprocessing.sequence.pad_sequences(
                            word_inputs,
                            padding='post'
        )
        dep_inputs = pad3d(dep_inputs)

        if len(self.target2idx) == 2:
            targets = np.array(targets, dtype='int32')
        else:
            targets = tf.keras.utils.to_categorical(
                        targets,
                        num_classes=len(self.target2idx),
                        dtype='int32'
            )

        return [word_inputs, dep_inputs], targets

    def make_embeddings(self):
        """
        Create an embedding matrix and a vocabulary dict for the dataset.

        Indexes 0 and 1 are reserved for the special <PAD> and <UNK> embeddings.
        """
        self.dataset_vecs = [
                np.zeros(self.embeddings.shape[1]),
                self.make_oov_vector(),
        ]
        self.dataset_voc = {
                '<PAD>': 0,
                '<UNK>': 1,
        }

        if self.make_embeddings_subset:
            for token in self.tokens:
                if token not in self.dataset_voc and token in self.token2idx:
                    self.dataset_vecs.append(
                        self.embeddings[self.token2idx[token]]
                    )
                    self.dataset_voc[token] = len(self.dataset_voc)
        else:
            for token, idx in self.token2idx.items():
                if token not in self.dataset_voc:
                    self.dataset_vecs.append(self.embeddings[idx])
                    self.dataset_voc[token] = len(self.dataset_voc)

        self.dataset_vecs = np.array(self.dataset_vecs, dtype=np.float32)


def pad3d(inputs, value=0, dtype='int32'):
    """Create a 3d-array with padded values from nested lists."""
    max_idx1 = max(len(lst) for lst in inputs)
    max_idx2 = max(len(lst2) for lst1 in inputs for lst2 in lst1)
    size = (len(inputs), max_idx1, max_idx2)
    padded = np.zeros(size, dtype=dtype) + value
    for i, seqs in enumerate(inputs):
        for j, seq in enumerate(seqs):
            padded[i, j, :len(seq)] = seq
    return padded
