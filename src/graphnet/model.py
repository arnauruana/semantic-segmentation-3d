import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MLP, DynamicEdgeConv


class DGCNN(nn.Module):
    """
    Dynamic Graph Convolutional Neural Network.

    Extends:
        nn.Module
    """

    def __init__(self, in_channels: int, out_channels: int, k: int) -> None:
        """
        Dynamic Graph Convolutional Neural Network.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            k (int): number of nearest neighbors.
        """
        super().__init__()

        self.conv1 = DynamicEdgeConv(nn=MLP([2 * in_channels, 64, 64]), k=k)
        self.conv2 = DynamicEdgeConv(nn=MLP([2 * 64, 64, 64]), k=k)
        self.conv3 = DynamicEdgeConv(nn=MLP([2 * 64, 64, 64]), k=k)

        self.mlp = MLP([3 * 64, 1024, 256, 128, out_channels], dropout=0.5, norm=None)

    def forward(self, graph: Data) -> torch.Tensor:
        """
        Computes the forward pass for the given graph.

        Args:
            graph (Data): homogenous graph to be forwarded.

        Returns:
            torch.Tensor: class probability tensor.
        """
        feats, coords = graph.x.float(), graph.pos.float()
        x0 = torch.cat([feats, coords], dim=-1)

        batch = graph.batch.float()
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)

        out = self.mlp(torch.cat([x1, x2, x3], dim=1))
        return F.log_softmax(out, dim=1)


if __name__ == "__main__":
    graph = Data(
        x=torch.rand(10, 1),
        y=torch.rand(10, 1),
        pos=torch.rand(10, 3),
        batch=torch.arange(10),
    )
    print(graph)
    network = DGCNN(in_channels=4, out_channels=3, k=3)
    print(network)
    probs = network(graph)
    print(probs)
