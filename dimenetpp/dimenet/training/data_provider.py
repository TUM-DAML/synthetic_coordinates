from collections import OrderedDict, defaultdict
import numpy as np
import tensorflow as tf
from .data_container import index_keys


class DataProvider:
    def __init__(self, data_containers, ntrain=np.inf, nvalid=np.inf, batch_size=1, seed=None,
                dist={'type': 'none'}):
        self.data_containers = data_containers
        # initialize with zeros
        self.nsamples = defaultdict(int)

        for key in data_containers:
            self.nsamples[key] = len(self.data_containers[key])

        self.batch_size = batch_size

        if ntrain is None:
            ntrain = np.inf
        if nvalid is None:
            nvalid = np.inf
            
        self.nsamples['train'] = min(self.nsamples['train'], ntrain)
        self.nsamples['val'] = min(self.nsamples['val'], nvalid)

        # Random state parameter, such that random operations are reproducible if wanted
        self._random_state = np.random.RandomState(seed=seed)

        # Store indices of training, validation and test data
        self.idx = {key: np.arange(self.nsamples[key]) for key in ['train', 'val', 'test']}

        # Index for retrieving batches
        self.idx_in_epoch = {'train': 0, 'val': 0, 'test': 0}

        # dtypes of dataset values
        self.dtypes_input = OrderedDict()
        self.dtypes_input['node_attr'] = tf.float32
        self.dtypes_input['edge_type'] = tf.int32
        self.dtypes_input['Dij'] = tf.float32
        self.dtypes_input['Anglesijk'] = tf.float32
        for key in index_keys:
            self.dtypes_input[key] = tf.int32
        self.dtype_target = tf.float32

        # Shapes of dataset values
        self.shapes_input = {}
        self.shapes_input['node_attr'] = [None, 14]
        self.shapes_input['edge_type'] = [None]

        # add extra dims for min, max dist/angle and center angle in case of rdkit
        extra_dim = []
        if dist['type'] == 'rdkit_bounds':
            extra_dim = [3]
        elif dist['type'] == 'ppr_rdkit_bounds':
            extra_dim = [4]
        self.shapes_input['Dij'] = [None] + extra_dim
        self.shapes_input['Anglesijk'] = [None] + extra_dim
        
        for key in index_keys:
            self.shapes_input[key] = [None]
        self.shape_target = [None, len(data_containers['test'][[0]]['targets'])]

    def shuffle_train(self):
        """Shuffle the training data"""
        self.idx['train'] = self._random_state.permutation(self.idx['train'])

    def get_batch_idx(self, split):
        """Return the indices for a batch of samples from the specified set"""
        start = self.idx_in_epoch[split]

        # Is epoch finished?
        if self.idx_in_epoch[split] == self.nsamples[split]:
            start = 0
            self.idx_in_epoch[split] = 0

        # shuffle training set at start of epoch
        if start == 0 and split == 'train':
            self.shuffle_train()

        # Set end of batch
        self.idx_in_epoch[split] += self.batch_size
        if self.idx_in_epoch[split] > self.nsamples[split]:
            self.idx_in_epoch[split] = self.nsamples[split]
        end = self.idx_in_epoch[split]

        return self.idx[split][start:end]

    def idx_to_data(self, split, idx, return_flattened=False):
        """Convert a batch of indices to a batch of data"""
        batch = self.data_containers[split][idx]

        if return_flattened:
            inputs_targets = []
            for key, dtype in self.dtypes_input.items():
                inputs_targets.append(tf.constant(batch[key], dtype=dtype))
            inputs_targets.append(tf.constant(batch['targets'], dtype=tf.float32))
            return inputs_targets
        else:
            inputs = {}
            for key, dtype in self.dtypes_input.items():
                inputs[key] = tf.constant(batch[key], dtype=dtype)
            targets = tf.constant(batch['targets'], dtype=tf.float32)
            return (inputs, targets)

    def get_dataset(self, split):
        """Get a generator-based tf.dataset"""
        def generator():
            while True:
                idx = self.get_batch_idx(split)
                yield self.idx_to_data(split, idx)
        return tf.data.Dataset.from_generator(
                generator,
                output_types=(dict(self.dtypes_input), self.dtype_target),
                output_shapes=(self.shapes_input, self.shape_target))

    def get_idx_dataset(self, split):
        """Get a generator-based tf.dataset returning just the indices"""
        def generator():
            while True:
                batch_idx = self.get_batch_idx(split)
                yield tf.constant(batch_idx, dtype=tf.int32)
        return tf.data.Dataset.from_generator(
                generator,
                output_types=tf.int32,
                output_shapes=[None])
