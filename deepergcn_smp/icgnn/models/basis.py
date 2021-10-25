'''
Basis function representations of distance and angle
'''
import torch
import numpy as np

def get_gaussian_basis(dist, num_basis, max_dist=None):
    if max_dist is None:
        # the largest distance
        max_dist = torch.max(dist)

    # n equally spaced bins between 0 and max
    centers = torch.linspace(0, max_dist, num_basis, dtype=dist.dtype)
    # the size of each bin
    std = centers[1] - centers[0]
    # insert a size "1" dimension
    return torch.exp(-0.5 * (dist.unsqueeze(-1) - centers.unsqueeze(0))**2 / std**2)

def get_bessel_basis(dist, num_basis):
    '''
    bessel function: dimenet paper eq 7
    https://arxiv.org/pdf/2003.03123.pdf
    
    dist: (N,) lengths of graph edges
    num_basis: int, number of basis functions

    return: (N, num_basis)
    '''
    
    # the largest distance
    c = torch.max(dist)
    n = torch.arange(1, num_basis + 1)

    # add a dimension to broadcast distance into each basis function
    dist = dist.unsqueeze(-1)
    basis = torch.sqrt(2 / c) * torch.sin(n * np.pi * dist / c) / dist
    
    return basis

# express the pairwise distance in terms of a basis of functions
def get_dist_basis(dist, type, num_basis, max_dist=None):
    if type == 'gaussian':
        return get_gaussian_basis(dist, num_basis, max_dist=max_dist)
    elif type == 'bessel':
        return get_bessel_basis(dist, num_basis)

# express the xyz angles in a similar basis
def get_cos_basis(angle, num_basis):
    k = torch.arange(num_basis, dtype=angle.dtype).unsqueeze(0)
    return torch.cos(k * angle.unsqueeze(-1))

