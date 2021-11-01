"""
Transform graph using the PPR as a distance
and angles based on this distance
"""

import numpy as np

import torch
import torch_geometric
from torch_geometric.data import Data
import torch_geometric.transforms as T

from ..models.embeddings import ppr
from ..models.basis import get_dist_basis, get_cos_basis
from ..data_utils.icgnn_dataset import ICGNN_Data


class Add_Linegraph(object):
    def __init__(self):
        pass

    def __call__(self, data):
        """
        data: a Data graph object

        return: ICGNN_Data object with the same graph
        """
        graph_copy = Data(edge_index=data.edge_index.clone(), num_nodes=data.num_nodes)
        transform = T.Compose([T.LineGraph(force_directed=True), T.Constant()])
        linegraph = transform(graph_copy)
        # remove self loops
        noloop_mask = ~(
            data.edge_index[0, linegraph.edge_index[0]]
            == data.edge_index[1, linegraph.edge_index[1]]
        )
        line_idx_noloop = data.edge_index.new_empty((2, noloop_mask.sum()))
        line_idx_noloop[0] = torch.masked_select(linegraph.edge_index[0], noloop_mask)
        line_idx_noloop[1] = torch.masked_select(linegraph.edge_index[1], noloop_mask)
        linegraph.edge_index = line_idx_noloop

        new_data = ICGNN_Data(
            x_g=data.x,
            edge_index_g=data.edge_index,
            edge_attr_g=data.edge_attr,
            edge_index_lg=linegraph.edge_index,
            y=data.y,
        )

        # transfer any other attributes to the new data object
        # such as distances, angles, ..
        # exclude the one which we already copied
        all_attrs = (
            "x",
            "edge_index",
            "edge_attr",
            "y",
            "distances",
            "edge_dist",
            "max_edge_dist",
            "min_edge_dist",
            "max_dist",
            "min_dist",
            "edge_dist_basis",
        )
        extra_attr = set(all_attrs) - set(["edge_index", "edge_attr", "y"])
        for attr in extra_attr:
            try:
                val = getattr(data, attr)
                setattr(new_data, attr, val)
            except AttributeError:
                pass

        return new_data


def nearest_psd_mat(A):
    """
    Find the nearest PSD matrix to a given matrix `A`

    Taken from: https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix/63131250#63131250

    return: matrix of same dimensions
    """
    C = (A + A.T) / 2
    eigval, eigvec = np.linalg.eigh(C)
    eigval[eigval < 0] = 0

    return eigvec.dot(np.diag(eigval)).dot(eigvec.T).real


def nearPSD(A, epsilon=0):
    """
    Find the nearest PSD matrix to a given matrix `A`
    taken from: https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix/63131250#63131250
    """
    n = A.shape[0]
    eigval, eigvec = np.linalg.eigh(A)
    val = np.matrix(np.maximum(eigval, epsilon))
    vec = np.matrix(eigvec)
    T = 1 / (np.multiply(vec, vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    out = B * B.T
    return out


class Set_Linegraph_EdgeAttr(object):
    """
    Set data.edge_attr_lg to constant
    """

    def __init__(self, edge_attr_val=1):
        self.edge_attr_val = edge_attr_val

    def __call__(self, data):
        """
        data: ICGNNData
        """
        num_edges = data.edge_index_lg.shape[1]
        data.edge_attr_lg = torch.ones(num_edges, 1)

        return data


class Set_PPR_Distance(object):
    """
    Take a regular graph with node attributes and adj matrix
    Find the PPR matrix for every pair of nodes

    Find distances using the kernel formula (see below) and set distances in the
    data attributes
    """

    def __init__(self, alpha=0.15, num_dist_basis=4, dist_basis_type="gaussian"):
        """
        alpha: restart param for PPR calculation
        """
        self.alpha = 0.15
        self.num_dist_basis = num_dist_basis
        self.dist_basis_type = dist_basis_type

    def __call__(self, data):
        """
        data: a single Data object, only graph.
        linegraph: if True, set data.x_lg, else set data.edge_attr

        return: ICGNN_Data object which includes the linegraph properties
        """
        # TODO: handle difference between edge_index and edge_index_g
        adj = torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index).tocsr()
        ppr_notpsd = ppr(adj, self.alpha, normalization="symmetric")
        ppr_psd = nearest_psd_mat(ppr_notpsd)

        ppr_mat = torch.Tensor(ppr_psd)
        # k(x, x) = diagonal of ppr matrix
        # reshape to column vector
        ppr_self = ppr_mat.diagonal().reshape((-1, 1))
        # Hilbert space representation of CPD kernel
        # Given the kernel k(x, x') = PPR(x, x')
        # Distance^2 = ||phi(x) - phi(x')||^2 = -k(x, x') + 0.5(k(x, x) + k(x', x'))
        dist_sqr = torch.clamp(-ppr_mat + 0.5 * (ppr_self + ppr_self.T), 0, 1)
        data.distances = torch.sqrt(dist_sqr)

        # (E, 2) edge index, each row is (i,j) = edge from i to j
        ndx = data.edge_index.T
        # get the distance for every edge
        edge_lengths = data.distances[ndx[:, 0], ndx[:, 1]]
        data.edge_dist = edge_lengths
        # convert to a basis function representation
        data.edge_dist_basis = get_dist_basis(
            edge_lengths,
            type=self.dist_basis_type,
            num_basis=self.num_dist_basis,
            max_dist=1,
        )

        return data


class Set_Graph_EdgeAttr_Distance(object):
    """
    Set data.edge_attr <- data.edge_dist_basis
    If edge_attr already exists, append it
    """

    def __call__(self, data):
        try:
            # if edge attr already exists, concatenate to it
            # expect this case most of the time for OGB-mol graphs
            old_attr = data.edge_attr.type(torch.FloatTensor)
            new_edge_attr = torch.cat([old_attr, data.edge_dist_basis], dim=-1)
        except AttributeError:
            # edge attr is None
            new_edge_attr = data.edge_dist_basis

        data.edge_attr = new_edge_attr

        return data


class Set_Linegraph_NodeAttr_Distance(object):
    """
    Set data.x_lg <- data.edge_dist_basis
    """

    def __call__(self, data):
        data.x_lg = data.edge_dist_basis
        return data


def angle_from_dists(lg_ndx, g_ndx, d1_arr, d2_arr, d3_mat):
    """
    lg_ndx: linegraph edge index transpose (E1, 2)
    g_ndx: graph edge index transposed (E2, 2)

    For each angle in the graph A-B-C, given the distances
    d1_arr = AB (edge of the graph) = Length of edges
    d2_arr = BC (edge of the graph) = Length of edges
    d3_mat = AC (might not be an edge in the graph)

    compute the angle ABC for each node of the linegraph

    """
    # cosine values of angles
    # formula: cosB = cosABC = AB^2 + BC^2 - AC^2 / (2 * AB * BC)
    # AB and BC are the sides adjacent to angle B.
    # These are actual edges present in the graph so we already have the lengths

    # the edge AC need not be in the graph. get the distance from the distance matrix
    start_node = g_ndx[lg_ndx[:, 0], 0]
    end_node = g_ndx[lg_ndx[:, 1], 1]

    d1 = d1_arr[lg_ndx[:, 0]]
    d2 = d2_arr[lg_ndx[:, 1]]
    d3 = d3_mat[start_node, end_node]

    cosine = (d1 ** 2 + d2 ** 2 - d3 ** 2) / (2 * d1 * d2)
    cosine = torch.clamp(cosine, -1, 1)
    angle = torch.acos(cosine)

    return angle


def get_min_basis(data, n_basis):
    """
    Get min angle basis
    """
    angle = angle_from_dists(
        data.edge_index_lg.T,
        data.edge_index_g.T,
        data.max_edge_dist,
        data.max_edge_dist,
        data.min_dist,
    )
    basis = get_cos_basis(angle, n_basis)

    return basis


def get_max_basis(data, n_basis):
    """
    Get max angle basis
    """
    angle = angle_from_dists(
        data.edge_index_lg.T,
        data.edge_index_g.T,
        data.min_edge_dist,
        data.min_edge_dist,
        data.max_dist,
    )
    basis = get_cos_basis(angle, n_basis)

    return basis


def get_center_basis(data, basis):
    """
    get the "center" angle by using the average of min and max distances
    over edges
    """
    mean_edge_dist = (data.max_edge_dist + data.min_edge_dist) / 2
    # all pairs
    mean_dist = (data.max_dist + data.min_dist) / 2

    angle = angle_from_dists(
        data.edge_index_lg.T,
        data.edge_index_g.T,
        mean_edge_dist,
        mean_edge_dist,
        mean_dist,
    )
    basis = get_cos_basis(angle, basis)

    return basis


def get_single_basis(data, n_basis):
    angle = angle_from_dists(
        data.edge_index_lg.T,
        data.edge_index_g.T,
        data.edge_dist,
        data.edge_dist,
        data.distances,
    )
    new_attr = get_cos_basis(angle, n_basis)

    return new_attr


class Set_Linegraph_EdgeAttr_Angle(object):
    """
    Get angles between edges using all-pairs node distances
    Store the angles as the linegraph edge attribute
    """

    def __init__(self, num_cos_basis=4, mode=None):
        """
        num_cos_basis: dimension of the angle embedding
        """
        if mode == "center_both":
            assert num_cos_basis >= 3, "Num basis should be atleast 3"

        self.num_cos_basis = num_cos_basis
        self.mode = mode

    def get_ppr_data(self, data):
        """
        create a data object with only PPR distances
        """
        return Data(
            edge_index_g=data.edge_index_g,
            edge_index_lg=data.edge_index_lg,
            edge_dist=data.edge_dist[0],
            distances=data.distances[0],
        )

    def get_rdkit_data(self, data):
        """
        create a data object with only rdkit distances
        """
        return Data(
            edge_index_g=data.edge_index_g,
            edge_index_lg=data.edge_index_lg,
            max_dist=data.max_dist,
            min_dist=data.min_dist,
            max_edge_dist=data.max_edge_dist,
            min_edge_dist=data.min_edge_dist,
        )

    def __call__(self, data):
        """
        data: ICGNN_Data
        """
        # special case: both PPR and bounds matrix?
        if type(data.distances) is list:
            ppr_data = self.get_ppr_data(data)
            ppr_basis = get_single_basis(ppr_data, self.num_cos_basis)

            rdk_data = self.get_rdkit_data(data)

            center_basis = get_center_basis(rdk_data, self.num_cos_basis // 3)
            min_basis = get_min_basis(rdk_data, self.num_cos_basis // 3)
            max_basis = get_max_basis(rdk_data, self.num_cos_basis // 3)

            new_attr = torch.cat((ppr_basis, center_basis, min_basis, max_basis), -1)

        elif self.mode == "min":
            new_attr = get_min_basis(data, self.num_cos_basis)
        elif self.mode == "max":
            new_attr = get_max_basis(data, self.num_cos_basis)
        elif self.mode == "both":
            # half the basis for min, half for max
            new_attr1 = get_min_basis(data, self.num_cos_basis // 2)
            new_attr2 = get_max_basis(data, self.num_cos_basis // 2)
            # join both
            new_attr = torch.cat([new_attr1, new_attr2], -1)
        elif self.mode == "center":
            new_attr = get_center_basis(data, self.num_cos_basis)
        elif self.mode == "center_both":
            # emb has 3 parts -
            # center angle + max angle + min angle
            # set num_cos_basis to multiple of 3
            center_basis = get_center_basis(data, self.num_cos_basis // 3)
            min_basis = get_min_basis(data, self.num_cos_basis // 3)
            max_basis = get_max_basis(data, self.num_cos_basis // 3)

            new_attr = torch.cat([center_basis, min_basis, max_basis], -1)
        elif self.mode is None:
            # single set of distances, such as PPR
            new_attr = get_single_basis(data, self.num_cos_basis)
        else:
            raise NotImplementedError

        data.edge_attr_lg = new_attr

        # all valid values?
        assert torch.isnan(data.edge_attr_lg).sum() == 0

        return data


class Remove_Distances(object):
    """
    remove distance matrices
    otherwise the attributes cant be collated
    """

    def __init__(self):
        pass

    def __call__(self, data):
        # unset unnecessary tensors that could have been set earlier
        data.distances = None
        data.max_dist, data.min_dist = None, None
        data.edge_dist = None
        data.max_edge_dist, data.min_edge_dist = None, None

        return data


class Detach(object):
    """
    Detach tensors that require grad
    """

    def __init__(self):
        pass

    def __call__(self, data):
        data.x_g = data.x_g.detach()
        return data
