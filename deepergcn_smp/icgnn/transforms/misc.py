import torch
import numpy as np

def get_radius_graph_from_lengths(edge_ndx, edge_lengths, cutoff=5):
    '''
    edge_ndx: current edge_ndx
    edge_lengths: length of each edge
    '''
    keep = edge_lengths < cutoff
    new_edge_ndx = edge_ndx[:, keep]

    return new_edge_ndx, keep

def get_radius_graph_from_distmat(dist, cutoff=5):
    '''
    dist: (N, N) matrix of distances between graph nodes
    edge_ndx: current edge_ndx
    cutoff: threshold distance for edges in same units as the distance

    return: (E', 2) new edge index of the graph
    mask: mask into old edge_ndx to indicate which edges have been selected
    '''
    # keep only 1 copy of the distances to avoid repeated edges
    tmp_dist = np.triu(dist.copy())
    # rest of the values become 0 - set them to inf to cutoff
    tmp_dist[tmp_dist == 0] = np.inf
    # apply the cutoff, get true/false
    keep = tmp_dist < cutoff
    # find indices of all true values and stack into (2, E)
    new_edge_ndx = np.vstack(np.nonzero(keep))

    return new_edge_ndx

class Finalize_Dist_Basis:
    def __init__(self):
        pass
    def __call__(self, data):
        if type(data.edge_dist_basis) is list:
            data.edge_dist_basis = torch.cat(data.edge_dist_basis, dim=-1)

        return data

def set_or_append(data, attr_name, new_attr):
    '''
    data: torch_geometric.data.Data
    attr_name: str
    new_attr: torch.Tensor

    if data has attr_name, make it a list (if required) and append new_attr
    else set it
    '''
    if hasattr(data, attr_name):
        # exists - make it a list and add new attrs
        old_attr = getattr(data, attr_name)
        if type(old_attr) is not list: 
            old_attr = [old_attr]
        setattr(data, attr_name, old_attr + [new_attr])
    else:
        # first time - not a list
        setattr(data, attr_name, new_attr)
    return data
