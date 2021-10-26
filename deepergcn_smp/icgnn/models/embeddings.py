import os, os.path as osp

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import torch_geometric
from ..data_utils.cache_utils import read_cache, write_cache
from .infomax import generate_dgi

EMB_DIR = "/nfs/students/yeshwant/datasets"

__all__ = ["generate_embedding", "create_embedding", "add_extra_features", "ppr"]


def add_extra_features(g_x, emb, feature_type):
    """
    g_x: graph.x (N x D1)
    emb: embedding for each node (N x D2)
    feature_type: 'norm', 'vector', 'unit-vector'
    """
    n1, _ = g_x.size()
    n2, _ = emb.size()

    assert n1 == n2

    if feature_type == "norm":
        extra = torch.norm(emb, dim=-1, keepdim=True)
    elif feature_type == "vector":
        extra = emb
    elif feature_type == "unit-vector":
        extra = emb / torch.norm(emb, dim=-1, keepdim=True)
    else:
        raise NotImplementedError

    return torch.cat((g_x, extra), dim=-1)


def create_embedding(graph, emb_dim, emb_type):
    # edges are in the COO (coordinate) format
    # each e = (2, 1) entry is an edge from vertex e[0, 0] -> e[1, 0]
    adj_matrix = torch_geometric.utils.to_scipy_sparse_matrix(graph.edge_index).tocsr()

    if emb_type == "verse":
        embedding = full_verse(adj_matrix, emb_dim)
    elif emb_type == "verse-l2":
        embedding = full_verse(adj_matrix, emb_dim, l2=True)
    elif emb_type == "dgi":
        embedding = generate_dgi(graph, emb_dim)
    else:
        raise NotImplementedError

    return embedding


def generate_embedding(dataset, graph, dataset_name, emb_type, emb_dim):
    """
    Read vertex embeddings from file or generate using the respective function

    graph: torch_geometric.data
    dataset_name: str
    emb_type: 'verse', 'verse-l2', 'dgi'
    emb_dim: dimension of the emb vector for each vertex
    """
    # the target/existing filename
    # one cache for each combination of (dataset, embedding dim, embedding type)
    filename = f"{dataset_name.lower()}_{emb_dim}_{emb_type}.pkl"
    cache_path = osp.join(EMB_DIR, filename)

    if os.path.isfile(cache_path):
        return read_cache(cache_path)

    print(f"Creating embedding: {emb_type}")
    embedding = create_embedding(graph, emb_dim, emb_type)
    write_cache(cache_path, embedding)

    return embedding


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


def full_verse(
    adj,
    emb_dim=64,
    similarity="ppr",
    ppr_alpha=0.85,
    learning_rate=0.005,
    max_epochs=2000,
    verbose=False,
    l2=False,
):
    """Generate node embeddings using F-VERSE.

    Args:
        adj: Adjacency matrix of the graph.
        embedding_dim: Dimension of the generated embeddings.
        similarity: Similarity measure to use. Supported values {'ppr', 'adj'}.
        ppr_alpha: alpha value for PPR
    """
    num_nodes = adj.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    emb = torch.normal(mean=0, std=1 / emb_dim, size=(num_nodes, emb_dim)).to(device)
    emb = nn.Parameter(emb)
    if verbose:
        print("Training verse with cuda:", emb.is_cuda)

    # target embedding
    if similarity == "ppr":
        target = ppr(adj, ppr_alpha)
    elif similarity == "adj":
        deg = np.ravel(adj.sum(1))
        target = adj / deg[:, None]
    else:
        raise NotImplementedError

    target_tensor = torch.Tensor(target).to(device)

    optimizer = torch.optim.Adam([emb], lr=learning_rate)

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        if l2:
            # optimize with l2 distance
            # take negative - high distance = low score
            scores = -pairwise_distance(emb, emb)
        else:
            # otherwise dot product
            scores = emb @ torch.transpose(emb, 0, 1)

        log_probs = F.log_softmax(scores, dim=-1)
        loss = F.kl_div(log_probs, target_tensor, reduction="sum")

        loss.backward()
        optimizer.step()

        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch:4d}, loss = {loss.item():.4f}")
    return emb.cpu().detach().numpy()
