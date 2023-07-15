import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data

sys.path.append(os.path.abspath("src/"))
from pointcloud import PointCloud


class Dataset(torch_geometric.data.Dataset):
    """
    Custom Dataset.

    Extends:
        torch_geometric.data.Dataset
    """

    def __init__(
        self,
        path: str = "data/",
        indices: list[int] = None,
        samples: int = -1,
        shuffle: bool = False,
        rotate: bool = False,
        pre_transform: T.BaseTransform = None,
        transform: T.BaseTransform = None,
    ) -> None:
        """
        Custom Dataset.

        Args:
            path (str, optional): dataset's path with its corresponding tensors. Defaults to "data/".
            indices (list[int], optional): List of indices from path to take.Defaults to None.
            samples (int, optional): dataset's number of samples. Defaults to -1.
            shuffle (bool, optional): whether or not to shuffle the dataset. Defaults to False.
            rotate (bool, optional): whether or not to rotate the dataset. Defaults to False.
            pre_transform (torch_geometric.transforms.BaseTransform, optional): dataset's pre-transformation. Defaults to None.
            transform (torch_geometric.transforms.BaseTransform, optional): dataset's transformation. Defaults to None.
        """
        super().__init__()

        self.path = Path(path)
        self.samples = samples
        self.shuffle = shuffle
        self.rotate = rotate
        self.pre_transform = pre_transform
        self.transform = transform

        self.splits = np.array(list(self.path.glob("*.pt")))
        if indices:
            self.splits = self.splits[indices]

    def len(self) -> int:
        """
        Lenght of the dataset.

        Returns:
            int: dataset's number of splits.
        """
        return len(self.splits)

    def get(self, idx: int) -> Data:
        """
        Gets elements of the dataset.

        Args:
            idx (int): dataset's indices.

        Returns:
            torch_geometric.data.Data: selected graph.
        """
        cloud = PointCloud(torch.load(self.splits[idx]))
        if self.samples != -1:
            cloud = cloud.sample(self.samples)
        if self.shuffle:
            cloud = cloud.shuffle()
        if self.rotate:
            cloud = cloud.rotate()

        graph = Data(
            x=cloud.feats(),
            y=cloud.labels(),
            pos=cloud.coords(),
        )
        if self.pre_transform:
            graph = self.pre_transform(graph)
        if self.transform:
            graph = self.transform(graph)
        return graph


if __name__ == "__main__":
    dataset = Dataset(
        samples=1000,
        shuffle=True,
        rotate=True,
        pre_transform=T.KNNGraph(k=5),
        transform=T.RandomJitter(1e-2),
    )
    print(dataset)
    print(dataset[0])
