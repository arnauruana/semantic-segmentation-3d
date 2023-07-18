import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("src/"))
from pointcloud import PointCloud
from pointnet import Dataset, Model


@torch.no_grad()
def infer(network, testset):
    network.eval()
    for idx, clouds in enumerate(testset):
        coords = torch.stack([cloud.coords() for cloud in clouds])
        coords = coords.transpose(2, 1)
        coords = coords.float()
        coords = coords.to(device)

        feats = torch.stack([cloud.feats() for cloud in clouds])
        feats = feats.transpose(2, 1)
        feats = feats.float()
        feats = feats.to(device)

        predictions, _, _ = network(coords, feats)
        predictions = torch.squeeze(predictions)
        predictions = predictions.float()
        predictions = predictions.to(device)

        cloud = PointCloud(torch.load(testset.dataset.splits[idx]))
        cloud.plot(name=f"pointnet_grth-{idx}")
        cloud.plot(name=f"pointnet_pred-{idx}", predictions=predictions)


def main(_):
    dataset = Dataset()
    indices_test = np.linspace(3, len(dataset) - 1, 16, dtype=np.uint)

    testset = DataLoader(
        dataset=Dataset(
            indices=list(indices_test),
            shuffle=False,
            rotate=False,
        ),
        batch_size=1,
        shuffle=False,
    )

    network = Model(PointCloud.NUM_CLASSES, feature_transform=True)
    network.load_state_dict(torch.load(path_model))
    network.to(device)

    infer(network, testset)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_model = Path("model/pointnet.pt")
    sys.exit(main(sys.argv))
