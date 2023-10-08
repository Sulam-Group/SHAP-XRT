from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset


def signal(d: int) -> Tensor:
    """
    A function that returns a cross of size `d \times d` pixels.

    Parameters:
    -----------
    d: int
        the size of the cross
    """
    x = torch.zeros((d, d))
    for i in range(1, d - 1):
        x[i, i] = 1
        x[i, d - i - 1] = 1
    return x


def noise(d: int, sigma: float) -> MultivariateNormal:
    """
    A function that returns random Gaussian noise of size `d \times d` pixels.

    Parameters:
    -----------
    d: int
        the size of the noise.
    sigma: float
        the standard deviation of the noise.
    """
    mu = torch.zeros(d**2)
    cov = sigma**2 * torch.eye(d**2)
    noise = MultivariateNormal(mu, cov)
    return noise


class Crosses(Dataset):
    """
    A class for generating the synthetic image dataset described in Sec. 4.2 of the paper.
    """

    def __init__(self, m: int, r: int, s: int, d: int, sigma: float) -> None:
        """
        Initialize the dataset.

        Parameters:
        -----------
        m: int
            the number of samples in the dataset.
        r: int
            the number of rows of patches in each sample.
        s: int
            the number of columns of patches in each sample.
        d: int
            the size of each patch in pixels.
        sigma: float
            the standard deviation of the noise.
        """
        self.l = int(m)
        self.n = r
        self.m = s
        self.d = d
        # define the signal pattern (a cross)
        self._signal = signal(d)
        self._signal = self._signal.flatten()
        # generate noisy patches
        self._noise = noise(d, sigma)
        self.data = self._noise.sample((self.l * self.n * self.m,))
        # generate binary patch labels so that the bag labels are balanced
        self.p = 1 - (1 / 2) ** (1 / (self.n * self.m))
        self.patch_labels = torch.bernoulli(
            self.p * torch.ones((self.l * self.n * self.m,))
        )
        self.labels = (
            self.patch_labels.unfold(0, self.n * self.m, self.n * self.m).sum(dim=1)
            >= 1
        ).float()
        # inject signal in chosen slices
        positives = self.patch_labels.nonzero()
        self.data[positives] += self._signal
        # reshape patches into images
        self.data = self.data.view(self.l, self.n * self.m, self.d**2)
        self.data = self.data.unfold(2, self.d, self.d)
        self.data = self.data.view(self.l, 1, self.n, self.m, self.d, self.d)
        self.data = self.data.permute(0, 1, 2, 4, 3, 5).contiguous()
        self.data = self.data.view(self.l, 1, self.n * self.d, self.m * self.d)
        # normalize data
        mu = self.data.flatten().mean()
        std = self.data.flatten().std()
        self.data = (self.data - mu) / std
        # reshape patch labels
        self.patch_labels = self.patch_labels.unfold(
            0, self.n * self.m, self.n * self.m
        )

    def __len__(self) -> int:
        """
        A function that returns the length of the dataset.
        """
        return self.l

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        A function that returns an itme in the dataset.

        Parameters:
        -----------
        idx: int
            the index of the item to return.
        """
        return self.data[idx], self.labels[idx]

    def cond(self, x: Tensor, C: set, M: int) -> Tensor:
        """
        A function that defines the conditional distribution to mask features.
        Gien a sample `x` and a set of features `C` to condition on, this
        function masks the features not in `C` with their conditional distribution.

        Parameters:
        -----------
        x: torch.Tensor
            the input sample to mask.
        C: set
            the set of features to condition on.
        M: int
            the number of samples to generate.
        """
        x_patch = (
            x.unfold(1, self.d, self.d)
            .unfold(2, self.d, self.d)
            .flatten(start_dim=1, end_dim=2)
        )

        not_C = set(range(x_patch.size(1))) - C
        R = self._noise.sample((M, len(not_C))).view(M, len(not_C), self.d, self.d)

        X = x_patch.repeat(M, 1, 1, 1)
        X[:, list(not_C)] = R
        X = X.view(M, x_patch.size(1), self.d**2)
        X = X.unfold(2, self.d, self.d)
        X = X.view(M, 1, self.n, self.m, self.d, self.d)
        X = X.permute(0, 1, 2, 4, 3, 5).contiguous()
        X = X.view(M, 1, self.n * self.d, self.m * self.d)
        return X


class BooleanDataset(Dataset):
    """
    A class for generating the Boolean dataset described in Sec. 4.1 of the paper.
    """

    def __init__(self, m: int, k: int, n: int) -> None:
        """
        Initialize the dataset.

        Parameters:
        -----------
        m: int
            the number of samples in the dataset.
        k: int
            the number of components in each sample.
        n: int
            the number of features in each component.
        """
        self.m = m
        self.k = k
        self.n = n
        # define signal and noise
        self._signal = torch.distributions.Normal(4, 1)
        self._noise = torch.distributions.Normal(0, 1)
        # generate noise
        self.data = self._noise.sample((m, n * k))
        # generate signal
        signal = self._signal.sample((k * m,))
        # define position of important features in disjunctions
        sample_idx = torch.arange(self.m).unsqueeze(1).repeat(1, self.k).view(-1)
        signal_idx = (
            (torch.randint(0, self.n, (m, k)) * self.k) + torch.arange(self.k)
        ).view(-1)
        # inject signal into samples
        self.data[sample_idx, signal_idx] = signal

    def __len__(self) -> int:
        """
        A function that returns the length of the dataset.
        """
        return self.m

    def __getitem__(self, idx: int) -> Tensor:
        """
        A function that returns an item in the dataset.

        Parameters:
        -----------
        idx: int
            the index of the item to return.
        """
        return self.data[idx]

    def cond(self, x: Tensor, C: set, M: int) -> Tensor:
        """
        A function that defines the conditional distribution to mask features.
        Given a sample `x` and a set of features `C` to condition on, this
        function masks the features not in `C` with their conditional distribution.

        Parameters:
        -----------
        x: torch.Tensor
            the input sample to mask.
        C: set
            the set of features to condition on.
        M: int
            the number of samples to generate.
        """
        not_C = set(range(x.size(1))) - C
        R = self._noise.sample((M, len(not_C)))
        x = x.repeat(M, 1)
        x[:, list(not_C)] = R
        return x
