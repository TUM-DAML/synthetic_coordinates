import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class BesselBasisLayer(layers.Layer):
    def __init__(self, num_radial, cutoff, name="bessel_basis", split=1, **kwargs):
        """
        split: split the basis into n parts. Input must have n channels, basis
                is computed over each of them, then all parts are concatenated.
        """

        super().__init__(name=name, **kwargs)
        self.num_radial = num_radial
        self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float32)
        self.split = split

        # Initialize frequencies at canonical positions
        def freq_init(shape, dtype):
            return tf.constant(
                np.pi * np.arange(1, shape + 1, dtype=np.float32), dtype=dtype
            )

        self.frequencies = self.add_weight(
            name="frequencies",
            shape=self.num_radial // split,
            dtype=tf.float32,
            initializer=freq_init,
            trainable=True,
        )

    def get_basis(self, dist):
        d_scaled = dist * self.inv_cutoff

        # Necessary for proper broadcasting behaviour
        d_scaled = tf.expand_dims(d_scaled, -1) + 1e-6

        return tf.sin(self.frequencies * d_scaled) / d_scaled

    def call(self, inputs):
        # 3 or 4 splits: input contains (ppr), min, max and center distances
        if self.split > 1:
            all_basis = []
            for split in range(self.split):
                all_basis.append(self.get_basis(inputs[:, split]))

            return tf.concat(all_basis, -1)
        # single distance
        else:
            return self.get_basis(inputs)
