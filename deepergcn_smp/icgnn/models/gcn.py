import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap

class GCNNetGraphCls(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCNNetGraphCls, self).__init__()

        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.lin1 = torch.nn.Linear(hidden_dim, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x1 = gap(x, batch)

        x = F.relu(self.conv2(x, edge_index))
        x2 = gap(x, batch)

        x = x1 + x2

        x = F.relu(self.lin1(x))
        x = F.log_softmax(x, dim=-1)

        return x

class GCNNet(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_features, 16, cached=True)
        self.conv2 = GCNConv(16, num_classes, cached=True)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        
    def forward(self, x, edge_index):
        out = F.relu(self.conv1(x, edge_index))
        out = F.dropout(out, training=self.training)
        out = self.conv2(out, edge_index)
        return F.log_softmax(out, dim=1)