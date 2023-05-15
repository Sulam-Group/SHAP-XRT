import torch
from dataset import get_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

dataset = get_dataset("train", mu=[0, 0, 0], std=[1, 1, 1])
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
x = []
for i, data in enumerate(tqdm(dataloader)):
    _x, _ = data
    x.append(_x)
x = torch.cat(x, dim=0)
mu = x.mean(dim=(0, 2, 3))
std = x.std(dim=(0, 2, 3))
print(mu, std)
