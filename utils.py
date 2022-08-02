import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from scipy.special import binom
from itertools import combinations
from model import SimpleCNN, SimpleFCN
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from typing import Tuple, Callable, Optional

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


def HRT(
    f: Callable[[Tensor], Tensor],
    dataset: Dataset,
    j: int,
    n: int,
    t: Optional[float] = None,
):
    """
    A function that implements the Holdout Randomization Test (HRT) for
    binary-classification, where the test statistic is the 01 error rate.

    Parameters:
    -----------
    f: Callable
        the model to test.
    dataset: torch.utils.data.Dataset
        the test dataset to compute the 01 error rate on.
    j: int
        the feature to test importance of.
    n: int
        the total number of features in a sample.
    t: Optiona[float]
        the test statistic. If `None`, it is computed before performing the test.
    """
    # initialize the dataloader for the test dataset
    dataloader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=True)

    # compute the test statistic if it is not provided
    if t is None:
        correct = 0
        for _, data in enumerate(dataloader):
            input, label = data

            input = input.to(dataset.ref.device)
            label = label.to(dataset.ref.device)

            output = f(input)
            prediction = output.argmax(dim=1)

            correct += torch.sum(prediction == label)
        t = 1 - correct / len(dataset)

    # compute the null statistic
    C = set(range(n)) - {j}
    correct = 0
    for _, data in enumerate(dataloader):
        input, label = data

        input = input.to(dataset.ref.device)
        label = label.to(dataset.ref.device)

        output = f(dataset.cond(input, C))
        prediction = output.argmax(dim=1)

        correct += torch.sum(prediction == label)
    t_tilde = 1 - correct / len(dataset)

    # compute the one-sided p-value
    p = t >= t_tilde
    print(
        f"j = {j}, H_0: f _||_ X_{set([j])} | X_{set(C)}, t = {100*t:.2f}%, t_tilde = {100*t_tilde:.2f}%, p = {int(p)}"
    )
    return p
