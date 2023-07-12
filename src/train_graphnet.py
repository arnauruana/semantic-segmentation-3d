import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as T
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torchmetrics import JaccardIndex

from graph import Dataset
from models.graphnet import DGCNN as GraphNet

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


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

    predicted_batch_oh = torch.stack(predicted_batch_oh).to(config["device"])
    label_batch_oh = torch.stack(label_batch_oh).to(config["device"])

    iou = jaccard(predicted_batch_oh, label_batch_oh)
    return accuracy, iou


def train(network, criterion, optimizer, scheduler, trainset, validset):
    def train_epoch():
        log_interval = max(len(trainset) // config["logs"], 1)
        network.train()
        for batch, clouds in enumerate(trainset):
            clouds = clouds.to(config["device"])

            optimizer.zero_grad()

            predictions = network(clouds)
            predictions = predictions.to(config["device"])

            loss = criterion(predictions, clouds.y)
            loss.backward()

            optimizer.step()

            if batch % log_interval == 0 and batch > 0:
                print(f"   - train loss: {loss.item():.4f}")

    @torch.no_grad()
    def valid_epoch():
        acc = 0
        total = 0
        network.eval()
        for _, clouds in enumerate(validset):
            clouds = clouds.to(config["device"])

            predictions = network(clouds)
            predictions = predictions.to(config["device"])

            acc, iou = correct_predictions(predictions, clouds.y, acc)
            total += 1

        total_points = total * config["samples"] * config["batch"]
        print(f"   - valid accu: {acc/total_points:.4f}")
        print(f"   - valid mIoU: {iou:.4f}")

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}:")
        train_epoch()
        valid_epoch()
        scheduler.step()
        name = Path(f"graphnet-s{config['samples']}-b{config['batch']}.pt")
        torch.save(network.state_dict(), path_model / name)


@torch.no_grad()
def test(network, testset):
    pass


def main(_):
    trainset, validset, testset = random_split(
        dataset=Dataset(
            path=path_data,
            samples=config["samples"],
            shuffle=True,
            rotate=True,
            pre_transform=T.KNNGraph(config["k"]),
            transform=T.RandomJitter(1e-2),
        ),
        lengths=[round(0.8 * 0.8, 2), round(0.8 * 0.2, 2), 0.2],
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

    network = GraphNet(Dataset.NUM_CLASSES, config["k"])
    network = network.to(config["device"])

    criterion = nn.NLLLoss(weight=class_weights.to(config["device"]))
    optimizer = optim.SGD(
        network.parameters(),
        lr=config["lr"],
        momentum=0.9,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=10,
        eta_min=0.0001,
    )

    train(network, criterion, optimizer, scheduler, trainset, validset)
    test(network, testset)

    name = Path(f"graphnet-s{config['samples']}-b{config['batch']}.pt")
    torch.save(network.state_dict(), path_model / name)


if __name__ == "__main__":
    path_data = Path("data/")
    path_model = Path("model/")

    config = {
        "batch": 2,
        "epochs": 10,
        "k": 100,
        "logs": 5,
        "lr": 0.1,
        "samples": 5000,
    }
    # config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["device"] = "cpu"

    jaccard = JaccardIndex(
        task="multiclass",
        num_classes=Dataset.NUM_CLASSES,
        average="weighted",
    ).to(config["device"])

    # class_weights = joblib.load("class_weights_ins.joblib")
    class_weights = joblib.load("class_weights_isns.joblib")
    class_weights = torch.FloatTensor(class_weights)
    class_weights = class_weights.to(config["device"])

    sys.exit(main(sys.argv))
