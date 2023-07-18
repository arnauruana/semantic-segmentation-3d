import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath("src/"))
from pointcloud import PointCloud


class Dataset(torch.utils.data.Dataset):
    """
    Custom Dataset for PointNet.

    Extends:
        torch.utils.data.Dataset
    """

    def __init__(
        self,
        path: str = "data/",
        indices: list[int] = None,
        samples: int = -1,
        shuffle: bool = False,
        rotate: bool = False,
    ) -> None:
        """
        Custom Dataset for PointNet.

        Args:
            path (str, optional): dataset's path with its corresponding tensors. Defaults to "data/".
            indices (list[int], optional): List of indices from path to take.Defaults to None.
            samples (int, optional): dataset's number of samples. Defaults to -1.
            shuffle (bool, optional): whether or not to shuffle the dataset. Defaults to False.
            rotate (bool, optional): whether or not to rotate the dataset. Defaults to False.
        """
        super().__init__()

        self.path = Path(path)
        self.samples = samples
        self.shuffle = shuffle
        self.rotate = rotate

        self.splits = np.array(list(self.path.glob("*.pt")))
        if indices:
            self.splits = self.splits[indices]

    def __len__(self) -> int:
        """
        Lenght of the dataset.

        Returns:
            int: dataset's number of splits.
        """
        return len(self.splits)

    def __getitem__(self, idx: int) -> PointCloud:
        """
        Gets elements of the dataset.

        Args:
            idx (int): dataset's indices.

        Returns:
            PointCloud: selected pointcloud.
        """
        cloud = PointCloud(torch.load(self.splits[idx]))
        if self.samples > 0:
            cloud = cloud.sample(self.samples)
        if self.shuffle:
            cloud = cloud.shuffle()
        if self.rotate:
            cloud = cloud.rotate()
        return cloud


if __name__ == "__main__":
    dataset = Dataset(
        samples=1000,
        shuffle=True,
        rotate=True,
    )
    print(dataset[0])
