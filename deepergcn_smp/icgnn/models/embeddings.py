import numpy as np
import scipy.sparse as sp
import torch


def ppr(adj, alpha=0.15, normalization="symmetric") -> np.ndarray:
    """Compute Personalized PageRank vectors for all nodes.

    Args:
        adj: Adjacency matrix of the graph.
        alpha: Damping parameter.
        normalization: 'right' - right multiply the degree inverse
                       'symmetric' - multiple by sq.root on both sides

    Returns:
        ppr: Matrix where each row contains the PPR vector for each node.

    """
    if sp.issparse(adj):
        adj = adj.toarray()
    elif isinstance(adj, np.ndarray):
        pass
    else:
        raise ValueError(
            f"adj must be a sparse matrix or numpy array (got {type(adj)} instead)"
        )
    eps = 1e-6
    deg = adj.sum(1) + eps
    deg_inv = np.power(deg, -1)

    num_nodes = adj.shape[0]
    if normalization == "right":
        M = np.eye(num_nodes) - (1 - alpha) * adj * deg_inv[:, None]
    elif normalization == "symmetric":
        deg_inv_root = np.power(deg_inv, 0.5)
        M = (
            np.eye(num_nodes)
            - (1 - alpha) * deg_inv_root[None, :] * adj * deg_inv_root[:, None]
        )

    return alpha * np.linalg.inv(M)


def pairwise_distance(x1, x2):
    """
    x1: (n, d) vectors
    x2: (n, d) vectors

    find distance between every pair of vectors.

    return (n, n)
    """
    x1 = torch.unsqueeze(x1, dim=0)
    x2 = torch.unsqueeze(x2, dim=0)

    distances = torch.cdist(x1, x2, p=2)
    distances = torch.squeeze(distances, dim=0)
    # take negative to invert the scale -> high distance = low score

    return distances
