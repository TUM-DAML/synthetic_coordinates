import sympy as sym
import tensorflow as tf
from tensorflow.keras import layers

from .basis_utils import bessel_basis, real_sph_harm


class SphericalBasisLayer(layers.Layer):
    def __init__(
        self,
        num_spherical,
        num_radial,
        cutoff,
        name="spherical_basis",
        split=1,
        **kwargs
    ):
        """
        split: split the basis into n parts. Input must have n channels, basis
                is computed over each of them, then all parts are concatenated.
        """
        super().__init__(name=name, **kwargs)

        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical

        self.split = split

        self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float32)

        # retrieve formulas
        # split basis: spherical // split
        self.bessel_formulas = bessel_basis(num_spherical, num_radial // self.split)
        # split basis: spherical // split
        self.sph_harm_formulas = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        # convert to tensorflow functions
        x = sym.symbols("x")
        theta = sym.symbols("theta")
        for i in range(num_spherical):
            if i == 0:
                first_sph = sym.lambdify(
                    [theta], self.sph_harm_formulas[i][0], "tensorflow"
                )(0)
                self.sph_funcs.append(lambda tensor: tf.zeros_like(tensor) + first_sph)
            else:
                self.sph_funcs.append(
                    sym.lambdify([theta], self.sph_harm_formulas[i][0], "tensorflow")
                )
            for j in range(num_radial // self.split):
                self.bessel_funcs.append(
                    sym.lambdify([x], self.bessel_formulas[i][j], "tensorflow")
                )

    def call(self, inputs):
        # shapes
        # d: (d,) or (d, 3)
        # Angles: (A,) or (A, 3)
        # id_reduce_ji: (A,)
        d, Angles, id_reduce_ji = inputs

        # tensor shapes for self.split == 3 written below
        # s = spherical, r = radial basis
        # stack or concat?
        join_func = tf.stack if self.split == 1 else tf.concat
        # (d, 3)
        d_scaled = d * self.inv_cutoff + 1e-6
        # (d, 3) x (s * r/3) times
        rbf = [f(d_scaled) for f in self.bessel_funcs]
        # (d, sr)
        rbf = join_func(rbf, axis=1)
        # (A, sr)
        rbf = tf.gather(rbf, id_reduce_ji)

        # (A,3) x s times
        cbf = [f(Angles) for f in self.sph_funcs]
        # (A, 3s)
        cbf = join_func(cbf, axis=1)
        # (A, sr)
        cbf = tf.repeat(cbf, self.num_radial // self.split, axis=1)

        return rbf * cbf
