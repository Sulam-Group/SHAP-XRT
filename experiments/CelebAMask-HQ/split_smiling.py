import os
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import get_dataset
from tqdm import tqdm


data_dir = "data"
dataset_dir = os.path.join(data_dir, "CelebAMask-HQ/celebamaskhq")

dataset = get_dataset("all")
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

attr_name = "Smiling"
attr_idx = dataset.attr_names.index(attr_name)

n = len(dataset)
target = []
for _, data in enumerate(tqdm(dataloader)):
    _, attr = data
    target.extend(attr[:, attr_idx].tolist())
target = torch.tensor(target)

idx = torch.arange(n)
pos_idx, neg_idx = idx[target == 1].tolist(), idx[target == 0].tolist()

ratio = 0.9
train_pos_size, train_neg_size = int(ratio * len(pos_idx)), int(ratio * len(neg_idx))

train_pos_idx = np.random.choice(pos_idx, train_pos_size, replace=False).tolist()
val_pos_idx = list(set(pos_idx) - set(train_pos_idx))

train_neg_idx = np.random.choice(neg_idx, train_neg_size, replace=False).tolist()
val_neg_idx = list(set(neg_idx) - set(train_neg_idx))

split = []
for i in range(n):
    if i in train_pos_idx or i in train_neg_idx:
        split.append([f"{i}.jpg", 0])
    elif i in val_pos_idx or i in val_neg_idx:
        split.append([f"{i}.jpg", 1])
    else:
        raise ValueError("Invalid index")

with open(os.path.join(dataset_dir, "CelebAMask-HQ-partition.txt"), "w") as f:
    writer = csv.writer(f, delimiter=" ")
    writer.writerows(split)
