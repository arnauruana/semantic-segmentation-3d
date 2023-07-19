import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex

sys.path.append(os.path.abspath("src/"))

from pointcloud import PointCloud
from pointnet import Dataset, Model


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
        coords = torch.stack([cloud.coords() for cloud in clouds])
        coords = coords.transpose(2, 1)
        coords = coords.float()
        coords = coords.to(device)

        feats = torch.stack([cloud.feats() for cloud in clouds])
        feats = feats.transpose(2, 1)
        feats = feats.float()
        feats = feats.to(device)

        predictions, _, _ = network(coords, feats)
        predictions = torch.reshape(
            predictions,
            (predictions.shape[0] * predictions.shape[1], PointCloud.NUM_CLASSES),
        )
        predictions = predictions.float()
        predictions = predictions.to(device)

        labels = torch.stack([cloud.labels() for cloud in clouds])
        labels = torch.squeeze(labels)
        labels = labels.float()
        labels = labels.to(device)

        hits, iou = correct_predictions(predictions, labels, hits)
        ious.append(iou)
        total += len(labels)

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
        ),
        batch_size=1,
        shuffle=False,
    )

    network = Model(PointCloud.NUM_CLASSES, feature_transform=True)
    network.load_state_dict(torch.load(path_model))
    network = network.to(device)

    test(network, testset)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_model = Path("model/pointnet.pt")

    jaccard = JaccardIndex(
        task="multiclass",
        num_classes=PointCloud.NUM_CLASSES,
        average="weighted",
    ).to(device)

    sys.exit(main(sys.argv))
