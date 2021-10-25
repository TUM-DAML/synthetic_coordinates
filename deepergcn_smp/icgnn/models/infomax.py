import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, DeepGraphInfomax

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

def train(model, graph, optimizer):
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, summary = model(graph.x, graph.edge_index)
    loss = model.loss(pos_z, neg_z, summary)
    loss.backward()
    optimizer.step()
    return loss.item()

def get_emb(model, graph):
    model.eval()
    z, _, _ = model(graph.x, graph.edge_index)
    return z

def generate_dgi(graph, emb_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model = DeepGraphInfomax(
        hidden_channels=emb_dim, encoder=Encoder(graph.num_features, emb_dim),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)

    graph = graph.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 300):
        loss = train(model, graph, optimizer)
        print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))

    emb = get_emb(model, graph)

    return emb.cpu().detach().numpy()


