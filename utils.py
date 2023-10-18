from itertools import combinations
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.special import binom
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from model import SimpleCNN, SimpleFCN
from shapxrt import shapxrt

# `wc` is the coefficient of each summand in the Shapley value
wc = lambda n, c: 1 / binom(n - 1, c) * 1 / n

# `models` contains the simple CNN and simple FCN models for the
# experiment on a synthetic image dataset in Sec. 4.2 of the paper
models = [
    ("CNN", SimpleCNN, torch.optim.Adam, 1e-03),
    ("FCN", SimpleFCN, torch.optim.SGD, 1e-02),
]

# hyper-parameters for the experiment on a synthetic
# image dataset in Sec. 4.2 of the paper
batch_size = 64
r = 2
s = 2
d = 7
alpha = 0.05


def train(
    train_dataloader: DataLoader,
    test_dataset: Dataset,
    net: nn.Module,
    optim: Optimizer,
    criterion: nn.Module,
) -> Tuple[nn.Module, float]:
    """
    A function to train a network on a binary-classification dataset.

    Parameters:
    -----------
    train_dataloader: torch.utils.data.DataLoader
        the dataloader for the training data
    test_dataset: torch.utils.data.Dataset
        the test dataset
    net: nn.Module
        the network to train
    optim: torch.optim.Optimizer
        the optimizer to use
    criterion: nn.Module
        the loss function to use
    """
    torch.set_grad_enabled(True)
    net.train()
    for data in train_dataloader:
        x, y = data

        output = net(x)
        pred = (output >= 0.5).float()
        loss = criterion(output, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

    torch.set_grad_enabled(False)
    net.eval()
    X = test_dataset.data
    Y = test_dataset.labels

    output = net(X)
    pred = (output >= 0.5).float()
    acc = (pred == Y).float().mean().item()

    torch.set_grad_enabled(True)
    return net, acc


def shaplit_power(net: nn.Module, test_dataset: Dataset, alpha: float, K: int) -> float:
    """
    A function to estimate the power of performing conditional independence
    testing through Shapley coefficient in the synthetic image dataset presented
    in Sec. 4.2 of the paper.

    Parameters:
    -----------
    net: nn.Module
        the network to test.
    test_dataset: torch.utils.data.Dataset
        the test dataset to perform the test over.
    alpha: float
        the significance level.
    K: int
        the number of null statistics to compute.
    """
    torch.set_grad_enabled(False)
    net.eval()

    X = test_dataset.data
    Y = test_dataset.labels

    # predict on the test dataset
    output = net(X)
    pred = (output >= 0.5).float()

    # find the true positive predictions
    TP_idx = (pred * Y).nonzero().squeeze()
    TP_X = X[TP_idx]
    TP_patch_Y = test_dataset.patch_labels[TP_idx]

    # `b` contains the bound `1 - \gamma_{j,C}`
    b = []
    # for each true positive prediction
    for x, x_patch_y in zip(TP_X, TP_patch_Y):
        N = set(range(len(x_patch_y)))
        # for each positive patch in the sample
        for j, y in enumerate(x_patch_y):
            y = int(y)
            if y == 0:
                continue

            S = N - {j}
            # for each subset of patches
            for c in range(len(S) + 1):
                CC = combinations(S, c)
                for C in CC:
                    C = list(C)
                    # to estimate the power of the test
                    # ignore tests where `C` contains a cross
                    if x_patch_y[C].any():
                        continue
                    C = set(C)
                    C.add(j)
                    f = net(test_dataset.cond(x, C, K))

                    C.remove(j)
                    f_null = net(test_dataset.cond(x, C, K))

                    g = (f - f_null).mean().item()
                    b.append(1 - g)

    # estimate the power of the test
    b = np.array(b)
    b = np.sort(b)
    ecdf = 1.0 * np.arange(len(b)) / (len(b) - 1)
    d = np.abs(b - alpha)
    i = np.argmin(d)
    p = ecdf[i]
    return p


def shapley(f, x, j, cond, M=1000):
    n = x.size(1)
    N = set(range(n))

    phi = 0
    S = N - {j}
    for c in range(len(S) + 1):
        _wc = wc(n, c)
        CC = combinations(S, c)
        for C in CC:
            C = set(C)
            C.add(j)
            f_cui = f(cond(x, C, M))

            C.remove(j)
            f_c = f(cond(x, C, M))

            phi += _wc * (f_cui - f_c).mean()
    return phi
