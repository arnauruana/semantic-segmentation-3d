import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torchmetrics import JaccardIndex

sys.path.append(os.path.abspath("src/"))
from graphnet import Dataset, Model
from pointcloud import PointCloud


def correct_predictions(predicted_batch, label_batch, accuracy):
    predicted_batch_oh = []
    label_batch_oh = []
    for batch_idx in range(predicted_batch.size()[0]):
        pred_idx = torch.argmax(predicted_batch[batch_idx])
        predicted_batch_oh.append(pred_idx)
        gt_idx = label_batch[batch_idx]
        label_batch_oh.append(gt_idx)
        if pred_idx == gt_idx:
            accuracy += 1

    predicted_batch_oh = torch.stack(predicted_batch_oh).to(device)
    label_batch_oh = torch.stack(label_batch_oh).to(device)

    iou = jaccard(predicted_batch_oh, label_batch_oh)
    return accuracy, iou


@torch.no_grad()
def test(network, testset):
    total, hits = 0, 0
    ious = []
    network.eval()
    for clouds in testset:
        clouds = clouds.to(device)

        predictions = network(clouds)
        predictions = predictions.to(device)

        hits, iou = correct_predictions(predictions, clouds.y, hits)
        ious.append(iou)
        total += len(clouds.x)

    accuracy = hits / total
    miou = torch.mean(torch.tensor(ious))
    print(f"   - test accu: {accuracy:.4f}")
    print(f"   - test mIoU: {miou:.4f}")


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
    network = network.to(device)

    test(network, testset)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_model = Path("model/graphnet.pt")
    knn = 100

    jaccard = JaccardIndex(
        task="multiclass",
        num_classes=PointCloud.NUM_CLASSES,
        average="weighted",
    ).to(device)

    sys.exit(main(sys.argv))
