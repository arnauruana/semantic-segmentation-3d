import os
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.express as px
import torch
from natsort import natsorted as sorted


class PointCloud(torch.Tensor):
    def __getitem__(self, idx):
        if isinstance(idx, str):
            idx = Dataset.FEATURES.index(idx)
            if self.dim() == 2:
                idx = (slice(None), idx)
        elif isinstance(idx, tuple):
            if isinstance(idx[1], str):
                if self.dim() == 1:
                    idx = [Dataset.FEATURES.index(i) for i in idx]
                elif self.dim() == 2:
                    idx = (idx[0], Dataset.FEATURES.index(idx[1]))
            elif isinstance(idx[1], slice) and isinstance(idx[1].start, str):
                idx = (
                    idx[0],
                    slice(
                        Dataset.FEATURES.index(idx[1].start),
                        Dataset.FEATURES.index(idx[1].stop) + 1,
                        idx[1].step,
                    ),
                )
        elif isinstance(idx, list) and isinstance(idx[0], str):
            idx = [Dataset.FEATURES.index(i) for i in idx]
            if self.dim() == 2:
                idx = (slice(None), idx)
        elif isinstance(idx, slice) and isinstance(idx.start, str):
            idx = slice(
                Dataset.FEATURES.index(idx.start),
                Dataset.FEATURES.index(idx.stop) + 1,
                idx.step,
            )
            if self.dim() == 2:
                idx = (slice(None), idx)
        return super().__getitem__(idx)

    def coordinates(self):
        return torch.Tensor(self["x":"z"])

    def coords(self):
        return self.coordinates()

    def features(self):
        return torch.Tensor(self[["reflectance", "echo", "num_echos"]])

    def feats(self):
        return self.features()

    def labels(self, one_hot=True):
        if one_hot:
            classes = [str(int(cls.item())) for cls in self["class"]]
            classes = [list(Dataset.COLORS).index(cls) for cls in classes]
            labels = np.zeros([len(self), Dataset.NUM_CLASSES])
            for idx, cls in enumerate(classes):
                labels[idx, cls] = 1
            return torch.from_numpy(labels)

        return torch.tensor(
            [list(Dataset.COLORS).index(str(int(cls.item()))) for cls in self["class"]],
        ).type(torch.long)

    def sample(self, samples):
        building = (self["class"] == 203000000).nonzero()
        road = (self["class"] == 202020000).nonzero()
        sidewalk = (self["class"] == 202030000).nonzero()

        num_building_ = len(building)
        num_road_ = len(road)
        num_sidewalk_ = len(sidewalk)
        num_total_ = num_building_ + num_road_ + num_sidewalk_

        wb = num_building_ / (num_total_)
        wr = num_road_ / (num_total_)
        ws = num_sidewalk_ / (num_total_)

        num_building = round((len(self) - samples) * wb)
        num_road = round((len(self) - samples) * wr)
        num_sidewalk = round((len(self) - samples) * ws)
        num_total = num_building + num_road + num_sidewalk

        if (
            0.8 * num_building_ > num_building
            and 0.8 * num_road_ > num_road
            and 0.8 * num_sidewalk_ > num_sidewalk
        ):
            if len(self) - num_total > samples:
                num_building += abs((len(self) - samples) - num_total)
            elif len(self) - num_total < samples:
                num_sidewalk -= abs((len(self) - samples) - num_total)

            idxs_building = set()
            idxs_road = set()
            idxs_sidewalk = set()

            while len(idxs_building) < num_building:
                idxs_building.add(np.random.randint(0, len(building)))

            while len(idxs_road) < num_road:
                idxs_road.add(np.random.randint(0, len(road)))

            while len(idxs_sidewalk) < num_sidewalk:
                idxs_sidewalk.add(np.random.randint(0, len(sidewalk)))

            samples_building = [
                building[idx_building] for idx_building in idxs_building
            ]
            samples_road = [road[idx_road] for idx_road in idxs_road]
            samples_sidewalk = [
                sidewalk[idx_sidewalk] for idx_sidewalk in idxs_sidewalk
            ]

            samples = samples_building + samples_road + samples_sidewalk

            data = self.numpy()
            data = np.delete(self.numpy(), samples, axis=0)
            data = PointCloud(torch.from_numpy(data))

        else:
            idxs = set()
            while len(idxs) < samples:
                idxs.add(np.random.randint(0, len(self)))
            data = self[list(idxs)]

        return data

    def shuffle(self):
        indexes = torch.randperm(len(self))
        return self[indexes]

    def rotate(self, angle=np.random.randint(0, 360), axis=(0, 0, 1)):
        def get_rotation_matrix(axis, angle):
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
            return rotation_matrix

        self[:, :3] = torch.matmul(
            input=self.coordinates(),
            other=get_rotation_matrix(axis, np.radians(-angle)).t(),
        )
        return self

    def plot(self, path, name="demo", save=True, show=False):
        if not save and not show:
            raise ValueError("Either 'save' or 'show' must be 'True'")

        fig = px.scatter_3d(
            x=self["x"],
            y=self["y"],
            z=self["z"],
            color=[str(int(label.item())) for label in self["class"]],
            color_discrete_map=Dataset.COLORS,
            title=f"PointCloud: {name}",
        )
        fig.update_traces(marker={"size": 2})

        if save:
            os.makedirs(path, exist_ok=True)
            time = str(datetime.timestamp(datetime.now())).split(".")[0]
            fig.write_html(path / Path(f"{name}__{time}.html"))
            fig.write_json(path / Path(f"{name}__{time}.json"))

        if show:
            fig.show()

        return True


class Dataset(torch.utils.data.Dataset):
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

    def __init__(self, path, samples=-1, shuffle=True, rotate=True):
        super().__init__()
        self.path = Path(path)
        self.splits = sorted(list(self.path.glob("*.pt")))
        self.samples = samples
        self.shuffle = shuffle
        self.rotate = rotate

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, idx):
        cloud = PointCloud(torch.load(self.splits[idx]))
        if self.samples > 0:
            cloud = cloud.sample(self.samples)
        if self.shuffle:
            cloud = cloud.shuffle()
        if self.rotate:
            cloud = cloud.rotate()
        return cloud


if __name__ == "__main__":
    path_data = Path("data/")
    path_split = Path("data/split_000_left.pt")
    path_plot = Path("plot/")

    equal = lambda x, y: x.tolist() == y.tolist()

    print("⏳", action := "testing pointcloud", end="\r")
    try:
        cloud = PointCloud(points := torch.load(path_split))

        assert type(points) == torch.Tensor, "error 1"
        assert type(cloud) == PointCloud, "error 2"

        assert equal(cloud[0], points[0]), "error 3"
        assert equal(cloud["x"], points[:, 0]), "error 4"
        assert equal(cloud[0][0], points[0][0]), "error 5"
        assert equal(cloud[0]["x"], points[0][0]), "error 6"
        assert equal(cloud[0, 0], points[0, 0]), "error 7"
        assert equal(cloud[0, "x"], points[0, 0]), "error 8"

        assert equal(cloud[[0, 1]], points[[0, 1]]), "error 9"
        assert equal(cloud[["x", "y"]], points[:, :2]), "error 10"

        assert equal(cloud[:2], points[:2]), "error 11"
        assert equal(cloud["x":"z"], points[:, :3]), "error 12"
        assert equal(cloud[0][:2], points[0][:2]), "error 13"
        assert equal(cloud[0]["x":"z"], points[0][:3]), "error 14"
        assert equal(cloud[0, :2], points[0, :2]), "error 15"
        assert equal(cloud[0, "x":"z"], points[0, :3]), "error 16"

        assert equal(cloud.coords(), points[:, :3]), "error 17"
        assert equal(cloud.feats(), points[:, [3, 6, 7]]), "error 18"
        assert equal(cloud.labels(False), points[:, 5]), "error 19"

        assert cloud.sample(10).shape == (10, 8), "error 20"
        assert not equal(cloud.sample(10000), cloud.sample(10000)), "error 21"

        assert cloud.shuffle().shape == cloud.shape, "error 22"
        assert not equal(cloud.shuffle(), cloud.shuffle()), "error 23"

        assert cloud.rotate().shape == cloud.shape, "error 24"
        assert cloud.plot(path_plot), "error 25"

    except AssertionError as error:
        print("❌", action, f"({error})")

    else:
        print("✔️", action)

    print("⏳", action := "testing dataset", end="\r")
    try:
        cloud = PointCloud(torch.load(path_split))

        dataset = Dataset(path_data, shuffle=False, rotate=False)
        assert type(dataset) == Dataset, "error 1"
        assert type(dataset[0] == PointCloud), "error 2"
        assert equal(dataset[0], cloud), "error 3"

        dataset = Dataset(path_data, samples=10)
        assert dataset[0].shape == (10, 8), "error 4"

        dataset_1 = Dataset(path_data, shuffle=False)
        dataset_2 = Dataset(path_data, shuffle=False)
        assert equal(dataset_1[0], dataset_2[0]), "error 5"
        dataset_1.shuffle = True
        dataset_2.shuffle = True
        assert not equal(dataset_1[0], dataset_2[0]), "error 6"

    except AssertionError as error:
        print("❌", action, f"({error})")

    else:
        print("✔️", action)
