import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics import JaccardIndex

from graph import Dataset
from models.pointnet import PointNetDenseCls as PointNet

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def correct_predictions(predicted_batch, label_batch, accuracy):
    predicted_batch_oh = []
    label_batch_oh = []
    for batch_idx in range(predicted_batch.size()[0]):
        pred_idx = torch.argmax(predicted_batch[batch_idx])
        predicted_batch_oh.append(pred_idx)
        gt_idx = torch.argmax(label_batch[batch_idx])
        label_batch_oh.append(gt_idx)
        if pred_idx == gt_idx:
            accuracy += 1

    predicted_batch_oh = torch.stack(predicted_batch_oh).to(config["device"])
    label_batch_oh = torch.stack(label_batch_oh).to(config["device"])

    iou = jaccard(predicted_batch_oh, label_batch_oh)
    return accuracy, iou


def train(network, criterion, optimizer, scheduler, trainset, validset):
    def train_epoch():
        log_interval = max(len(trainset) // config["logs"], 1)
        network.train()
        for batch, clouds in enumerate(trainset):
            coords = torch.stack([cloud.coords() for cloud in clouds])
            coords = coords.transpose(2, 1)
            coords = coords.float()
            coords = coords.to(config["device"])

            feats = torch.stack([cloud.feats() for cloud in clouds])
            feats = feats.transpose(2, 1)
            feats = feats.float()
            feats = feats.to(config["device"])

            optimizer.zero_grad()

            predictions, _, _ = network(coords, feats)
            predictions = predictions.transpose(2, 1)
            predictions = predictions.float()
            predictions = predictions.to(config["device"])

            labels = torch.stack([cloud.labels() for cloud in clouds])
            labels = labels.transpose(2, 1)
            labels = labels.float()
            labels = labels.to(config["device"])

            loss = criterion(predictions, labels)
            loss.backward()

            optimizer.step()

            if batch > 0 and batch % log_interval == 0:
                print(f"  - train loss: {loss.item():.4f}")

    @torch.no_grad()
    def valid_epoch():
        network.eval()
        acc = 0
        for batch, clouds in enumerate(validset):
            coords = torch.stack([cloud.coords() for cloud in clouds])
            coords = coords.transpose(2, 1)
            coords = coords.float()
            coords = coords.to(config["device"])

            feats = torch.stack([cloud.feats() for cloud in clouds])
            feats = feats.transpose(2, 1)
            feats = feats.float()
            feats = feats.to(config["device"])

            predictions, _, _ = network(coords, feats)
            predictions = torch.reshape(
                predictions,
                (predictions.shape[0] * predictions.shape[1], 21),
            )
            predictions = predictions.float()
            predictions = predictions.to(config["device"])

            labels = torch.stack([cloud.labels() for cloud in clouds])
            labels = torch.reshape(
                labels,
                (labels.shape[0] * labels.shape[1], 21),
            )
            labels = labels.float()
            labels = labels.to(config["device"])

            acc, iou = correct_predictions(predictions, labels, acc)

            if batch % (len(validset) / config["batch"]) == 0 and batch != 0:
                print(
                    f"  - valid accuracy: {acc/((batch)*clouds.shape[0]*clouds.shape[1]) * 100:.2f}%"
                )
                print(f"  - valid acc: {acc}")
                print(f"  - valid IoU: {iou}")

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}:")
        train_epoch()
        valid_epoch()
        scheduler.step()


@torch.no_grad()
def test(network, testset):
    return


def main(_):
    trainset, validset, testset = random_split(
        dataset=Dataset(path_data, samples=config["samples"]),
        lengths=[round(0.9 * 0.8, 2), round(0.9 * 0.2, 2), 0.1],
    )

    trainset = DataLoader(
        dataset=trainset,
        batch_size=config["batch"],
        shuffle=True,
    )
    validset = DataLoader(
        dataset=validset,
        batch_size=config["batch"],
        shuffle=False,
    )
    testset = DataLoader(
        dataset=testset,
        batch_size=1,
        shuffle=False,
    )

    network = PointNet(Dataset.NUM_CLASSES, feature_transform=True)
    network = network.to(config["device"])

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config["device"]))
    optimizer = optim.Adam(network.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=config["step"],
        gamma=config["gamma"],
    )

    train(network, criterion, optimizer, scheduler, trainset, validset)
    test(network, testset)


if __name__ == "__main__":
    path_data = Path("data/")
    path_model = Path("model/")
    config = {
        "batch": 4,
        "epochs": 10,
        "gamma": 0.8,
        "logs": 5,
        "lr": 1e-3,
        "samples": 10000,
        "step": 30,
    }
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    jaccard = JaccardIndex(
        task="multiclass",
        num_classes=Dataset.NUM_CLASSES,
        average="weighted",
    ).to(config["device"])
    class_weights = torch.FloatTensor(
        [
            0.08076548143432968,
            0.1792716367264818,
            0.6232488827817019,
            1.5441436153835046,
            2.7883138279580804,
            9.016897203928629,
            9.349475450120504,
            14.747437665387052,
            28.582155317504462,
            30.67947276603993,
            37.55692550091593,
            60.115844487321674,
            139.0576348620629,
            146.89950252450254,
            228.11512740689497,
            246.30469965764084,
            282.4933961590633,
            417.17290458618874,
            471.4506136065769,
            546.1538992408557,
            1317.6440226440227,
        ]
    )
    print(config)
    sys.exit(main(sys.argv))
