from pathlib import Path

import torch
import torch_geometric
import torch_geometric.transforms as T
from natsort import natsorted as sorted
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from dataset import PointCloud


class Dataset(torch_geometric.data.Dataset):
    NUM_CLASSES = 21
    FEATURES = [
        "x",
        "y",
        "z",
        "reflectance",
        "id",
        "class",
        "echo",
        "num_echos",
    ]
    COLORS = {
        "203000000": "#808080",
        "202020000": "#000000",
        "202030000": "#C0C0C0",
        "303040200": "#0000FF",
        "304020000": "#00FF00",
        "202040000": "#708090",
        "202010000": "#DCDCDC",
        "303030302": "#F08080",
        "302020300": "#000000",
        "302030400": "#DEB887",
        "303020200": "#FFFF00",
        "304040000": "#008000",
        "303020300": "#FFFF00",
        "303020600": "#FFFF00",
        "302020400": "#BC8F8F",
        "301000000": "#FFEFD5",
        "303030202": "#FF69B4",
        "302020600": "#8A2BE2",
        "302021000": "#A52A2A",
        "303020000": "#FFFF00",
        "302020900": "#7FFFD4",
    }

    def __init__(
        self,
        path,
        samples=-1,
        shuffle=True,
        rotate=True,
        pre_transform=None,
        transform=None,
    ):
        super().__init__()
        self.path = Path(path)
        self.splits = sorted(list(self.path.glob("*.pt")))
        self.samples = samples
        self.shuffle = shuffle
        self.rotate = rotate
        self.pre_transform = pre_transform
        self.transform = transform

    def len(self):
        return len(self.splits)

    def get(self, idx):
        cloud = PointCloud(torch.load(self.splits[idx]))
        if self.samples > 0:
            cloud = cloud.sample(self.samples)
        if self.shuffle:
            cloud = cloud.shuffle()
        if self.rotate:
            cloud = cloud.rotate()

        graph = Data(
            x=cloud.feats(),
            y=cloud.labels(one_hot=False),
            pos=cloud.coords(),
        )
        if self.pre_transform:
            graph = self.pre_transform(graph)
        if self.transform:
            graph = self.transform(graph)
        return graph


if __name__ == "__main__":
    pre_transform = T.KNNGraph(k=30)
    transform = T.Compose([T.RandomJitter(1e-2)])
    dataset = Dataset(
        path="data/",
        samples=100,
        shuffle=False,
        rotate=False,
        pre_transform=pre_transform,
        transform=transform,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
    )
    print(next(iter(loader)))
