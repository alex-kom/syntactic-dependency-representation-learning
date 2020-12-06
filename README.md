# Syntactic Dependency Representation Learning

Train distributed representations of words and sentences from dependency syntax
trees like those described in <a id="1">[1]</a>.

The file format for the dependency parsed sentences is the following:\
Each line represents a node in the dependency tree and consists of
three space separated fields:\
`word dependency_type head_index` \
A `None` head index indicates the root node, and empty lines are sentence separators.\
Examples of parsed sentences in this format can be found in `TrecQC/trecqc_dep_trees.txt`.

## Dependency-based embeddings

`DepFeatureGenerator` provides generators to extract co-occurring tokens
from tree files to train word and dependency feature embeddings
with typical word embedding implementations.
For example, `DepFeatureGenerator.node_context_pair_generator`
extracts the `(target, context)` tokens to train an extended dependency-based
skip-gram model <a id="1">[1]</a>.
Alternatively, `DepFeatureGenerator.node_context_bag_generator`
generates token sequences, each containing the co-occurring tokens within
a dependency node's context neighborhood.
These can be used as pseudo-sentences
with word embedding implementations operating on sequences,
by setting their window parameter to a large enough value.

## Dependency-based sentence classification

Dependency feature embeddings can be used with word embeddings to
provide syntactic information to a sentence classifier.
`DependencyLSTM` is a configurable LSTM with attention
that operates on dependency parsed sentences.
An example usage of training and evaluating `DependencyLSTM`
on the *TREC Question Classification* dataset can be found in `run_trecqc_evaluation.py` 
Dependency parsed sentences and a sub-set of low precision pre-trained dependency embeddings
for running the experiment are included in the `TrecQC` folder.

## References
<a id="1">[1]</a>
Alexandros Komninos and Suresh Manandhar. 
Dependency based embeddings for sentence classification tasks.
In proceedings of NAACL, 2016.
