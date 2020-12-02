from itertools import permutations


class DepNode:
    """Container for dependency node attributes."""
    def __init__(self, token, dep, head_idx, dependent_indexes=None):
        self.token = token
        self.dep = dep
        try:
            self.head_idx = int(head_idx)
        except:
            self.head_idx = None
        self.dependent_indexes = dependent_indexes if dependent_indexes else []


class DepFeatureGenerator:
    """
    Generators of word and dependency features for dependency based models.

    Dependency trees are represented as array-like containers of DepNode
    objects. This representation preserves both the original order of tokens
    in a sentence and the graph structure of the dependency parsed sentences.

    The generators can be used to create (target, context) pairs or bags of
    co-occurring tokens to train word2vec style models.

    Format of the parsed sentences files:
    Each line corresponds to a dependency tree node and consists of three space
    separated fields: token dependency_type index_of_head.
    Empty lines are treated as sentence separators.

    Arguments:

        dep_trees : iterable of array-like containers of DepNode objects or None
            The parsed dependency trees. If None, dep_trees will be a generator
            yielding trees from the files specified in fnames.

        fnames : str or iterable of str.
            The filepaths containing the dependency parsed sentences.

        encoding : str
            Text encoding for reading the dependency parsed sentence files.

        n_iter : int
            The number of iterations reading the dependency parsed sentences.

        win_size : int
            The number of words before and after a token for generating
            contexts with sequence context type generators.

        words : bool
            Whether to return word tokens in dependency context type generators.

        deps : bool
            Whether to return dependency tokens in dependency context type
            generators.
    """
    def __init__(self,
                 dep_trees=None,
                 fnames=None,
                 encoding='utf8',
                 n_iter=1,
                 win_size=2,
                 words=True,
                 deps=True):
        self.fnames = [fnames] if isinstance(fnames, str) else fnames
        self.n_iter = n_iter
        self.encoding = encoding
        self.win_size = win_size
        self.words = words
        self.deps = deps
        self.dep_trees = (dep_trees if dep_trees is not None
                          else self.trees_from_files_generator())

    def get_trees(self):
        """Return a list containing the dependency tree objects."""
        return list(self.dep_trees)

    def trees_from_files_generator(self):
        """Generator of dependency tree objects from text files."""
        for iteration in range(self.n_iter):
            for fname in self.fnames:
                tree = []
                with open(fname, encoding=self.encoding) as file:
                    for line in file:
                        sep = line.split()
                        if sep:
                            tree.append(DepNode(*sep))
                        else:
                            add_dependent_indexes(tree)
                            yield tree
                            tree = []

    def node_context_bag_generator(self):
        """Generator of lists of tokens within node neighborhoods."""
        for tree in self.dep_trees:
            for node in tree:
                if self.words:
                    yield get_word_context(node, tree)
                if self.deps:
                    yield get_dep_context(node, tree)

    def node_context_pair_generator(self):
        """Generator of (target, context) tokens within node neighborhoods."""
        for tree in self.dep_trees:
            for node in tree:
                if self.words:
                    for pair in get_pairs(get_word_context(node, tree)):
                        yield pair
                if self.deps:
                    for pair in get_pairs(get_dep_context(node, tree)):
                        yield pair

    def window_context_pair_generator(self):
        """Generator of (target, context) tokens within window neighborhoods."""
        for tree in self.dep_trees:
            for idx, node in enumerate(tree):
                for token in get_window_context(idx, tree, self.win_size):
                    yield node.token, token

    def window_context_bag_generator(self):
        """Generator of lists of tokens within window neighborhoods."""
        for tree in self.dep_trees:
            for idx, node in enumerate(tree):
                yield [node.token] + get_window_context(
                        idx, tree, self.win_size
                        )

    def word_sequence_generator(self):
        """Generator of lists of word tokens for each sentence."""
        for tree in self.dep_trees:
            yield [node.token for node in tree]


def get_pairs(items):
    """Return all ordered pairs of tokens within items."""
    return permutations(items, 2)

def add_dependent_indexes(tree):
    """Fill the dependent_indexes list of each DepNode in the tree."""
    for idx, node in enumerate(tree):
        if node.head_idx is not None:
            tree[node.head_idx].dependent_indexes.append(idx)

def get_word_context(node, tree):
    """Return a list of word tokens within a node's neighborhood."""
    context_bag = [node.token]
    context_bag += [tree[idx].token for idx in node.dependent_indexes]
    if node.head_idx is not None:
        context_bag.append(tree[node.head_idx].token)
    return context_bag

def get_dep_context(node, tree):
    """Return a list of dependency tokens within a node's neighborhood."""
    context_bag = [node.token]
    context_bag += ['_r_'.join((tree[idx].dep, tree[idx].token)) for idx
                    in node.dependent_indexes]
    if node.head_idx is not None:
        context_bag.append('_'.join((node.dep, tree[node.head_idx].token)))
    return context_bag

def get_window_context(idx, tree, size):
    """Return a list of words within a 2*size window around the idx position."""
    return [node.token for node in
            tree[max(0, idx-size) : idx] + tree[idx+1 : idx+size+1]]

def print_tree(tree):
    """Print the contents of a tree."""
    for idx, node in enumerate(tree):
        print(idx, node.token, node.dep, node.head_idx, node.dependent_indexes)
