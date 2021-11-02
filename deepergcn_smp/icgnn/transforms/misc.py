import torch


class Finalize_Dist_Basis:
    def __init__(self):
        pass

    def __call__(self, data):
        if type(data.edge_dist_basis) is list:
            data.edge_dist_basis = torch.cat(data.edge_dist_basis, dim=-1)

        return data


def set_or_append(data, attr_name, new_attr):
    """
    data: torch_geometric.data.Data
    attr_name: str
    new_attr: torch.Tensor

    if data has attr_name, make it a list (if required) and append new_attr
    else set it
    """
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
