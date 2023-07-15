import os
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torchmetrics import JaccardIndex

sys.path.append(os.path.abspath("src/"))
from graphnet import Dataset, Model
from pointcloud import PointCloud
from utils import MAX_UINT_VALUE, MIN_UINT_VALUE

seed = random.randint(MIN_UINT_VALUE, MAX_UINT_VALUE)
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
        max_miou = 0
        hits = 0
        total = 0
        ious = []
        network.eval()
        for _, clouds in enumerate(validset):
            clouds = clouds.to(config["device"])

            predictions = network(clouds)
            predictions = predictions.to(config["device"])

            hits, iou = correct_predictions(predictions, clouds.y, hits)
            ious.append(iou)
            total += len(clouds.x)

        accuracy = hits / total
        miou = torch.mean(torch.tensor(ious))
        print(f"   - valid accu: {accuracy:.4f}")
        print(f"   - valid mIoU: {miou:.4f}")

        name = Path(f"graphnet-s{config['samples']}-k{config['k']}")
        os.makedirs(path_model, exist_ok=True)
        os.makedirs(path_model / name, exist_ok=True)
        if miou > max_miou:
            max_miou = miou
            torch.save(network.state_dict(), path_model / name / Path("best_ckpt.pt"))
            np.savetxt(
                fname=path_model / name / Path("metrics.txt"),
                X=[accuracy, miou],
            )

        torch.save(network.state_dict(), path_model / name / Path("last_ckpt.pt"))

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}:")
        train_epoch()
        if epoch % 2 == 0:
            valid_epoch()
        scheduler.step()


def main(_):
    dataset = Dataset()

    num_splits = 16
    indices_valid = np.linspace(0, len(dataset) - 4, num_splits, dtype=np.uint)
    indices_test = np.linspace(3, len(dataset) - 1, num_splits, dtype=np.uint)
    indices_train = np.arange(0, len(dataset))
    indices_train = set(indices_train) - set(indices_valid) - set(indices_test)

    trainset = DataLoader(
        dataset=Dataset(
            indices=list(indices_train),
            samples=config["samples"],
            shuffle=True,
            rotate=True,
            pre_transform=T.KNNGraph(config["k"]),
            transform=T.RandomJitter(1e-2),
        ),
        batch_size=config["batch"],
        shuffle=True,
    )
    validset = DataLoader(
        dataset=Dataset(
            indices=list(indices_valid),
            shuffle=False,
            rotate=False,
            pre_transform=T.KNNGraph(config["k"]),
        ),
        batch_size=1,
        shuffle=False,
    )
    testset = DataLoader(
        dataset=Dataset(
            indices=list(indices_test),
            shuffle=False,
            rotate=False,
            pre_transform=T.KNNGraph(config["k"]),
        ),
        batch_size=1,
        shuffle=False,
    )

    network = Model(
        in_channels=len(PointCloud.COLUMNS["features"])
        + len(PointCloud.COLUMNS["coordinates"]),
        out_channels=PointCloud.NUM_CLASSES,
        k=config["k"],
    )
    network = network.to(config["device"])

    criterion = nn.NLLLoss(weight=class_weights)
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


if __name__ == "__main__":
    path_model = Path("model/")

    config = {
        "batch": 8,
        "epochs": 10,
        "k": 30,
        "logs": 5,
        "lr": 0.1,
        "samples": 10000,
    }
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["seed"] = seed

    jaccard = JaccardIndex(
        task="multiclass",
        num_classes=PointCloud.NUM_CLASSES,
        average="weighted",
    ).to(config["device"])

    print(config)

    class_weights = joblib.load("class_weights_isns.joblib")
    class_weights = torch.FloatTensor(class_weights).to(config["device"])

    sys.exit(main(sys.argv))
