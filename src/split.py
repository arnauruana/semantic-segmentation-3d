import pandas as pd
import torch

FILE = "data_preprocessed_good_version.csv"


def my_norm(a, coord):
    assert coord in [1, 2], "Coord argument must be 1 or 2"
    if coord == 2:
        a = (((a - a.min()) / (a.max() - a.min())) * coord) - 1
    else:
        a = (a - a.min()) / (a.max() - a.min())
    return a


data = pd.read_csv(FILE)

splitted_data = []
for i in range(199):
    index = data.loc[(data["y"] >= i) & (data["y"] < 2 + i)].index

    splitted = data.loc[index]
    x_outlier = splitted["x"].quantile(q=[0.05, 0.95]).values
    index = splitted.loc[
        (splitted["x"] > x_outlier[0]) & (splitted["x"] < x_outlier[1])
    ].index

    splitted = data.loc[index]
    x_range = splitted["x"].quantile(q=[0.25, 0.75]).values
    x_mid = (x_range[1] - x_range[0]) / 2
    index_right = splitted.loc[(splitted["x"] > x_range[1] - x_mid)].index
    index_left = splitted.loc[(splitted["x"] <= x_range[1] - x_mid)].index

    splitted_data.append((data.loc[index_left], data.loc[index_right]))


for i, (left, right) in enumerate(splitted_data):
    left = left.iloc[:, 1:-1]
    left["x"] = my_norm(left["x"].values, 2)
    left["y"] = my_norm(left["y"].values, 2)
    left["z"] = my_norm(left["z"].values, 1)
    left = torch.tensor(left[left.columns].values)

    right = right.iloc[:, 1:-1]
    right["x"] = my_norm(right["x"].values, 2)
    right["y"] = my_norm(right["y"].values, 2)
    right["z"] = my_norm(right["z"].values, 1)
    right = torch.tensor(right[right.columns].values)

    path_left = f"data/split_{'00' if i < 10 else '0' if i < 100 else ''}{i}_left.pt"
    path_right = f"data/split_{'00' if i < 10 else '0' if i < 100 else ''}{i}_right.pt"

    torch.save(left, path_left)
    torch.save(right, path_right)
