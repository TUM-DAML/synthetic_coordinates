import tensorflow as tf
from tensorflow.keras import layers

from ..initializers import GlorotOrthogonal


class EmbeddingBlock(layers.Layer):
    def __init__(self, emb_size, activation=None, name="embedding", **kwargs):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size
        self.weight_init = GlorotOrthogonal()

        self.dense_node = layers.Dense(
            self.emb_size,
            activation=activation,
            use_bias=True,
            kernel_initializer=self.weight_init,
        )
        self.dense_edge = layers.Dense(
            self.emb_size,
            activation=activation,
            use_bias=True,
            kernel_initializer=self.weight_init,
        )
        self.dense_rbf = layers.Dense(
            self.emb_size,
            activation=activation,
            use_bias=True,
            kernel_initializer=self.weight_init,
        )
        self.dense = layers.Dense(
            self.emb_size,
            activation=activation,
            use_bias=True,
            kernel_initializer=self.weight_init,
        )

    def call(self, inputs):
        node_attr, edge_emb, rbf, idnb_i, idnb_j = inputs

        edge_emb = self.dense_edge(edge_emb)
        rbf = self.dense_rbf(rbf)

        node_emb = self.dense_node(node_attr)
        node_emb_i = tf.gather(node_emb, idnb_i)
        node_emb_j = tf.gather(node_emb, idnb_j)

        x = tf.concat([node_emb_i, node_emb_j, edge_emb, rbf], axis=-1)
        x = self.dense(x)
        return x
