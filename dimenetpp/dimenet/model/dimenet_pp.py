import numpy as np
import tensorflow as tf

from .layers.embedding_block import EmbeddingBlock
from .layers.bessel_basis_layer import BesselBasisLayer
from .layers.spherical_basis_layer import SphericalBasisLayer
from .layers.interaction_pp_block import InteractionPPBlock
from .layers.output_pp_block import OutputPPBlock
from .activations import swish


class DimeNetPP(tf.keras.Model):
    """
    DimeNet++ model.

    Parameters
    ----------
    emb_size
        Embedding size used for the messages
    out_emb_size
        Embedding size used for atoms in the output block
    int_emb_size
        Embedding size used for interaction triplets
    basis_emb_size
        Embedding size used inside the basis transformation
    num_blocks
        Number of building blocks to be stacked
    num_spherical
        Number of spherical harmonics
    num_radial
        Number of radial basis functions
    cutoff
        Cutoff distance for interatomic interactions
    num_before_skip
        Number of residual layers in interaction block before skip connection
    num_after_skip
        Number of residual layers in interaction block after skip connection
    num_dense_output
        Number of dense layers for the output blocks
    num_targets
        Number of targets to predict
    activation
        Activation function
    output_init
        Initialization method for the output layer (last layer in output block)
    """

    def __init__(
        self,
        emb_size,
        out_emb_size,
        int_emb_size,
        basis_emb_size,
        num_blocks,
        num_spherical,
        num_radial,
        cutoff=5.0,
        num_before_skip=1,
        num_after_skip=2,
        num_dense_output=3,
        num_targets=12,
        activation=swish,
        output_init="zeros",
        name="dimenet",
        dist={"type": "none"},
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_blocks = num_blocks

        # split the basis?
        dist_to_nsplits = {
            # only 1 distance
            "ppr": 1,
            # 3 distances from rdkit: min, max, center
            "rdkit_bounds": 3,
            # +ppr
            "ppr_rdkit_bounds": 4,
        }
        bes_split = dist_to_nsplits[dist["type"]]
        sph_split = bes_split

        # Cosine basis function expansion layer
        self.rbf_layer = BesselBasisLayer(num_radial, cutoff=cutoff, split=bes_split)
        self.sbf_layer = SphericalBasisLayer(
            num_spherical, num_radial, cutoff=cutoff, split=sph_split
        )

        # Edge type embeddings: 4 edge types: single, double, triple, and aromatic
        emb_init = tf.initializers.RandomUniform(minval=-np.sqrt(3), maxval=np.sqrt(3))

        self.edge_embeddings = tf.keras.layers.Embedding(
            4,
            emb_size,
            embeddings_initializer=emb_init,
        )

        # Embedding and first output block
        self.output_blocks = []
        self.emb_block = EmbeddingBlock(emb_size, activation=activation)
        self.output_blocks.append(
            OutputPPBlock(
                emb_size,
                out_emb_size,
                num_dense_output,
                num_targets,
                activation=activation,
                output_init=output_init,
            )
        )

        # Interaction and remaining output blocks
        self.int_blocks = []
        for _ in range(num_blocks):
            self.int_blocks.append(
                InteractionPPBlock(
                    emb_size,
                    int_emb_size,
                    basis_emb_size,
                    num_before_skip,
                    num_after_skip,
                    activation=activation,
                )
            )
            self.output_blocks.append(
                OutputPPBlock(
                    emb_size,
                    out_emb_size,
                    num_dense_output,
                    num_targets,
                    activation=activation,
                    output_init=output_init,
                )
            )

    def call(self, inputs):
        node_attr, edge_type = inputs["node_attr"], inputs["edge_type"]
        Dij, Anglesijk = inputs["Dij"], inputs["Anglesijk"]
        batch_seg = inputs["batch_seg"]
        idnb_i, idnb_j = inputs["idnb_i"], inputs["idnb_j"]
        id_expand_kj, id_reduce_ji = inputs["id_expand_kj"], inputs["id_reduce_ji"]
        n_atoms = tf.shape(node_attr)[0]

        # Embed edges, distances, and angles
        edge_emb = self.edge_embeddings(edge_type)
        rbf = self.rbf_layer(Dij)
        sbf = self.sbf_layer([Dij, Anglesijk, id_reduce_ji])

        # Embedding block
        x = self.emb_block([node_attr, edge_emb, rbf, idnb_i, idnb_j])
        P = self.output_blocks[0]([x, idnb_i, n_atoms])

        # Interaction blocks
        for i in range(self.num_blocks):
            x = self.int_blocks[i]([x, edge_emb, rbf, sbf, id_expand_kj, id_reduce_ji])
            P += self.output_blocks[i + 1]([x, idnb_i, n_atoms])

        P = tf.math.segment_sum(P, batch_seg)
        return P
