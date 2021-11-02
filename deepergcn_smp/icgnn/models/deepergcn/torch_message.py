from torch_geometric.nn import MessagePassing


class GenMessagePassing(MessagePassing):
    def __init__(self):
        super(GenMessagePassing, self).__init__(aggr="mean")

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return super(GenMessagePassing, self).aggregate(inputs, index, ptr, dim_size)
