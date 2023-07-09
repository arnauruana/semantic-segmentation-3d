import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, DynamicEdgeConv


class GraphNet(nn.Module):
    def __init__(self, out_channels, k=30, aggr="max"):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 6, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)

        self.mlp = MLP([3 * 64, 1024, 256, 128, out_channels], dropout=0.5, norm=None)

    def forward(self, data):
        batch_size, num_points = data.shape[0], data.shape[1]

        data = torch.reshape(data, (batch_size * num_points, -1))

        x = data.feats()
        x = x.float()
        x = x.cuda() if data.is_cuda else x.cpu()

        pos = data.coords()
        pos = pos.float()
        pos = pos.cuda() if data.is_cuda else pos.cpu()

        batch = torch.ones(len(data))
        for i in range(batch_size):
            batch[num_points * i : num_points + num_points * i] = i
        batch.float()
        batch = batch.cuda() if data.is_cuda else batch.cpu()

        x0 = torch.cat([x, pos], dim=-1)
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)

        out = self.mlp(torch.cat([x1, x2, x3], dim=1))
        return F.log_softmax(out, dim=1)
