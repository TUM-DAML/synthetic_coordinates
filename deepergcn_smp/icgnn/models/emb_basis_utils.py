"""
Utils to embed the distance and angle basis inside the mode
"""

import torch.nn as nn


def get_local_emb_layer(
    emb_basis_global, emb_basis_local, emb_bottleneck, attr_dim, emb_dim
):
    """
    embed the attr locally
    """
    if emb_basis_local:
        # has the basis been embedded to the bottleneck globally?
        if emb_basis_global:
            # then do bottleneck->hidden
            emb_edge_attr = nn.Linear(emb_bottleneck, emb_dim)
        else:
            # need to embed the basis here
            if emb_bottleneck:
                # basis->bottleneck->hidden
                emb_edge_attr = nn.Sequential(
                    nn.Linear(attr_dim, emb_bottleneck),
                    nn.Linear(emb_bottleneck, emb_dim),
                )
            else:
                # basis->hidden
                emb_edge_attr = nn.Linear(attr_dim, emb_dim)
    else:
        # basis already embedded to emb_dim
        emb_edge_attr = nn.Identity()

    return emb_edge_attr


def get_global_emb_layer(
    emb_basis_global, emb_basis_local, emb_bottleneck, attr_dim, emb_dim
):
    """
    embed the attr globally
    """
    if emb_basis_global:
        if emb_bottleneck:
            if emb_basis_local:
                # basis->emb_bottleneck
                global_emb_edge_attr = nn.Linear(attr_dim, emb_bottleneck)
            else:
                # basis->bottleneck->hidden_channels
                global_emb_edge_attr = nn.Sequential(
                    nn.Linear(attr_dim, emb_bottleneck),
                    nn.Linear(emb_bottleneck, emb_dim),
                )
        else:
            # basis->hidden_channels
            global_emb_edge_attr = nn.Linear(attr_dim, emb_dim)
    else:
        global_emb_edge_attr = nn.Identity()
    return global_emb_edge_attr
