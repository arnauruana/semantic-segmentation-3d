import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

sys.path.append(os.path.abspath("src/"))
from graphnet import Dataset, Model
from pointcloud import PointCloud


@torch.no_grad()
def infer(network, testset):
    network.eval()
    for idx, graph in enumerate(testset):
        graph = graph.to(device)

        prediction = network(graph)
        prediction = prediction.to(device)

        cloud = PointCloud(torch.load(testset.dataset.splits[idx]))
        cloud.plot(name=f"graphnet_grth-{idx}")
        cloud.plot(name=f"graphnet_pred-{idx}", predictions=prediction)


def main(_):
    dataset = Dataset()
    indices_test = np.linspace(3, len(dataset) - 1, 16, dtype=np.uint)

    testset = DataLoader(
        dataset=Dataset(
            indices=list(indices_test),
            shuffle=False,
            rotate=False,
            pre_transform=T.KNNGraph(knn),
        ),
        batch_size=1,
        shuffle=False,
    )

    network = Model(
        in_channels=len(PointCloud.COLUMNS["features"])
        + len(PointCloud.COLUMNS["coordinates"]),
        out_channels=PointCloud.NUM_CLASSES,
        k=knn,
    )
    network.load_state_dict(torch.load(path_model))
    network.to(device)

    infer(network, testset)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_model = Path("model/graphnet.pt")
    knn = 30
    sys.exit(main(sys.argv))
