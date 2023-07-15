import os
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.express as px
import torch


class PointCloud(torch.Tensor):
    """
    PointCloud.

    Extends:
        torch.Tensor
    """

    NUM_CLASSES = 11
    """Number of classes of the pointcloud."""

    COLUMN_NAMES = [
        "x",
        "y",
        "z",
        "reflectance",
        "id",
        "class",
        "echo",
        "num_echos",
    ]
    """Ordered column names of the pointcloud."""

    COLUMNS = {
        "coordinates": ["x", "y", "z"],
        "features": ["reflectance", "echo", "num_echos"],
        "labels": {
            "semantic": "class",
            "instance": "id",
            "panoptic": ["class", "id"],
        },
    }
    """Column associations per type."""

    COLOR_SCHEME = {
        "203000000": "#808080",
        "202020000": "#000000",
        "202030000": "#C0C0C0",
        "303040200": "#0000FF",
        "304020000": "#00FF00",
        "202040000": "#708090",
        "202010000": "#DCDCDC",
        "303030302": "#F08080",
        "303020200": "#FFA500",
        "302020300": "#DEB887",
        "302030400": "#FFFF00",
    }
    """Ordered color scheme per class of the pointcloud."""

    def __getitem__(self, idx):
        """
        Gets elements of the pointcloud.

        Args:
            idx (Any): pointcloud's indices

        Returns:
            Self: pointcloud's subset of elements
        """
        if isinstance(idx, str):
            idx = PointCloud.COLUMN_NAMES.index(idx)
            if self.dim() == 2:
                idx = (slice(None), idx)
        elif isinstance(idx, tuple):
            if isinstance(idx[1], str):
                if self.dim() == 1:
                    idx = [PointCloud.COLUMN_NAMES.index(i) for i in idx]
                elif self.dim() == 2:
                    idx = (idx[0], PointCloud.COLUMN_NAMES.index(idx[1]))
            elif isinstance(idx[1], slice) and isinstance(idx[1].start, str):
                idx = (
                    idx[0],
                    slice(
                        PointCloud.COLUMN_NAMES.index(idx[1].start),
                        PointCloud.COLUMN_NAMES.index(idx[1].stop) + 1,
                        idx[1].step,
                    ),
                )
        elif isinstance(idx, list) and isinstance(idx[0], str):
            idx = [PointCloud.COLUMN_NAMES.index(i) for i in idx]
            if self.dim() == 2:
                idx = (slice(None), idx)
        elif isinstance(idx, slice) and isinstance(idx.start, str):
            idx = slice(
                PointCloud.COLUMN_NAMES.index(idx.start),
                PointCloud.COLUMN_NAMES.index(idx.stop) + 1,
                idx.step,
            )
            if self.dim() == 2:
                idx = (slice(None), idx)
        return super().__getitem__(idx)

    def coordinates(self) -> torch.Tensor:
        """
        Gets the coordinates of the pointcloud.

        Returns:
            torch.Tensor: pointcloud's coordinates (x,y,z).
        """
        columns = PointCloud.COLUMNS["coordinates"]
        return torch.Tensor(self[columns]).float()

    def coords(self) -> torch.Tensor:
        """
        Gets the coordinates of the pointcloud.

        Returns:
            torch.Tensor: pointcloud's coordinates (x,y,z).
        """
        return self.coordinates()

    def features(self) -> torch.Tensor:
        """
        Gets the features of the pointcloud.

        Returns:
            torch.Tensor: pointcloud's features.
        """
        columns = PointCloud.COLUMNS["features"]
        return torch.Tensor(self[columns]).float()

    def feats(self) -> torch.Tensor:
        """
        Gets the features of the pointcloud.

        Returns:
            torch.Tensor: pointcloud's features.
        """
        return self.features()

    def labels(self, one_hot: bool = False) -> torch.Tensor:
        """
        Gets the labels of the pointcloud.

        Args:
            one_hot (bool, optional): whether or not to return the results in one-hot encoding. Defaults to False.

        Returns:
            torch.Tensor: pointcloud's labels.
        """
        column = PointCloud.COLUMNS["labels"]["semantic"]
        classes = [str(int(cls.item())) for cls in self[column]]
        classes = [list(PointCloud.COLOR_SCHEME).index(cls) for cls in classes]

        if one_hot:
            labels = np.zeros([len(self), PointCloud.NUM_CLASSES])
            for idx, cls in enumerate(classes):
                labels[idx, cls] = 1
            return torch.from_numpy(labels).long()

        return torch.Tensor(classes).long()

    def sample(self, samples: int):
        """
        Samples the pointcloud randomly.

        Args:
            samples (int): number of random samples (points) to be chosen.

        Returns:
            Self: pointcloud's subset of samples.
        """
        assert samples > 0, "samples > 0"
        assert samples <= len(self), f"samples <= {len(self)}"

        idxs = set()
        while len(idxs) < samples:
            idxs.add(np.random.randint(0, len(self)))
        return self[list(idxs)]

    def shuffle(self):
        """
        Shuffles the pointcloud randomly.

        Returns:
            Self: shuffled pointcloud.
        """
        return self[torch.randperm(len(self))]

    def rotate(
        self,
        angle: int = np.random.randint(0, 360),
        axis: tuple = (0, 0, 1),
    ):
        """
        Rotates the pointcloud for a given angle and axis.

        Args:
            angle (int, optional): rotation angle. Defaults to np.random.randint(0, 360).
            axis (tuple, optional): rotation axis. Defaults to (0, 0, 1).

        Returns:
            Self: _description_
        """

        def get_rotation_matrix(angle: float, axis: tuple) -> torch.Tensor:
            """
            Gets the rotation matrix for a given angle and axis.

            Args:
                angle (float): rotation angle in radians.
                axis (tuple): rotation axis.

            Returns:
                torch.Tensor: rotated pointcloud.
            """
            x, y, z = axis
            c = torch.cos(torch.tensor(angle))
            s = torch.sin(torch.tensor(angle))
            t = 1 - c
            rotation_matrix = torch.tensor(
                [
                    [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
                    [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
                    [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
                ]
            )
            return rotation_matrix.float()

        self[:, :3] = torch.matmul(
            input=self.coordinates(),
            other=get_rotation_matrix(np.radians(-angle), axis).t(),
        )
        return self

    def plot(
        self,
        path: str = "plot/",
        name: str = "demo",
        save: bool = True,
        show: bool = False,
    ) -> None:
        """
        Plots the pointcloud in a 3d space.

        Args:
            path (Path): save path
            name (str, optional): figure's name. Defaults to "demo".
            save (bool, optional): whether or not to save the results in memory. Defaults to True.
            show (bool, optional): whether or not to show the results immediately after generating (can be unstable). Defaults to False.
        """
        assert save == True or show == True, "save and show are false"

        column = PointCloud.COLUMNS["labels"]["semantic"]
        figure = px.scatter_3d(
            x=self["x"],
            y=self["y"],
            z=self["z"],
            color=[str(int(label.item())) for label in self[column]],
            color_discrete_map=PointCloud.COLOR_SCHEME,
            title=f"POINTCLOUD - {name.upper()} - {len(self)} POINTS",
        )
        figure.update_traces(marker={"size": 2})

        if save:
            os.makedirs(path, exist_ok=True)
            time = str(datetime.timestamp(datetime.now())).split(".")[0]
            figure.write_json(Path(path) / Path(f"{name}__{time}.json"))
            figure.write_html(Path(path) / Path(f"{name}__{time}.html"))

        if show:
            figure.show()


if __name__ == "__main__":
    cloud = PointCloud(torch.load("data/split_000_left.pt"))
    print(cloud)
    print(cloud.shape)

    cloud = cloud.sample(5000)
    print(cloud.shape)

    coords = cloud.coords()
    print(coords)
    feats = cloud.feats()
    print(feats)
    labels = cloud.labels()
    print(labels)

    cloud = cloud.shuffle()
    cloud = cloud.rotate()
    cloud.plot("plot")
