import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.distributions import MultivariateNormal, Normal, Uniform
from torchvision.datasets import ImageFolder
from typing import Tuple, Callable, Optional


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


class CrossesDataset(Dataset):
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
        self._signal = Normal(4, 1)
        self._noise = Normal(0, 1)
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


class BBBC041Dataset(ImageFolder):
    """
    A class to load the BBBC041 dataset used in Sec. 4.2 of the paper.
    """

    def __init__(
        self,
        root: str,
        ref: Optional[Tensor] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the dataset.

        Parameters:
        -----------
        root: str
            the folder containing the dataset.
        ref: Optional[Tensor]
            the reference value used to mask features.
        transform: Optional[Callable]
            a function/transform that takes in an PIL image and returns a transformed version.
        """
        super(BBBC041Dataset, self).__init__(root, transform)
        self.ref = ref
        if self.ref is not None:
            self.h, self.w = self.ref.size(1), self.ref.size(2)
            self.ref_patch = (
                self.ref.unfold(1, self.h // 2, self.h // 2)
                .unfold(2, self.w // 2, self.w // 2)
                .flatten(start_dim=1, end_dim=2)
            )

    def cond(self, x: Tensor, C: set) -> Tensor:
        """
        A function that defines the (un)conditional distribution to mask features.
        Given a batch of samples `x` and a set of features `C` to condition on, this
        function masks the features not in `C` with their conditional distribution.

        Parameters:
        -----------
        x: torch.Tensor
            the input batch to mask.
        C: set
            the set of features to condition on.
        """
        if self.ref is None:
            raise ValueError("Missing reference value to mask features.")

        x_patch = (
            x.unfold(2, self.h // 2, self.h // 2)
            .unfold(3, self.w // 2, self.w // 2)
            .flatten(start_dim=2, end_dim=3)
        )
        not_C = list(set(range(4)) - C)
        x_patch[:, :, not_C] = self.ref_patch[:, not_C]

        x = x_patch.view(
            x.size(0), x.size(1), x_patch.size(2), self.h // 2 * self.w // 2
        )
        x = x.unfold(3, self.w // 2, self.w // 2)
        x = x.view(x.size(0), x.size(1), 2, 2, self.h // 2, self.w // 2)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(x.size(0), x.size(1), 2 * self.h // 2, 2 * self.w // 2)
        return x


class SimpleSigmoidDataset(Dataset):
    def __init__(self, m):
        self.m = m

        mu1 = mu2 = 0.0
        sigma1 = sigma2 = 5.0
        self._x1, self._x2 = Uniform(-6, 6), Uniform(-6, 6)

        x1, x2 = self._x1.sample((self.m,)), self._x2.sample((self.m,))
        self.data = torch.stack((x1, x2), dim=1)

    def __len__(self):
        return self.m

    def __getitem__(self, idx):
        return self.data[idx]

    def cond(self, x, C, M):
        _x1, _x2 = x
        if 0 not in C:
            x1 = self._x1.sample((M,))
        if 0 in C:
            x1 = _x1.repeat((M,))
        if 1 not in C:
            x2 = self._x2.sample((M,))
        if 1 in C:
            x2 = _x2.repeat((M,))

        x = torch.stack((x1, x2), dim=1)
        return x


class SigmoidDataset(Dataset):
    def __init__(self, m: int):
        self.m = m

        mu1 = mu2 = 0.0
        mu3 = mu4 = 0.0
        rho1, rho2 = 0.2, -0.2
        sigma1 = sigma2 = 3.0
        sigma3 = sigma4 = 3.0
        cov1, cov2 = torch.tensor(
            [
                [sigma1**2, rho1 * sigma1 * sigma2],
                [rho1 * sigma1 * sigma2, sigma2**2],
            ]
        ), torch.tensor(
            [
                [sigma3**2, rho2 * sigma3 * sigma4],
                [rho2 * sigma3 * sigma4, sigma4**2],
            ]
        )

        self._x1x2, self._x3x4 = MultivariateNormal(
            torch.tensor([mu1, mu2]), cov1
        ), MultivariateNormal(torch.tensor([mu3, mu4]), cov2)

        self._cond1 = lambda x2: Normal(
            mu1 + sigma1 / sigma2 * rho1 * (x2 - mu2), (1 - rho1**2) * sigma1**2
        )
        self._cond2 = lambda x1: Normal(
            mu2 + sigma2 / sigma1 * rho1 * (x1 - mu1), (1 - rho1**2) * sigma2**2
        )
        self._cond3 = lambda x4: Normal(
            mu3 + sigma3 / sigma4 * rho2 * (x4 - mu4), (1 - rho2**2) * sigma3**2
        )
        self._cond4 = lambda x3: Normal(
            mu4 + sigma4 / sigma3 * rho2 * (x3 - mu3), (1 - rho2**2) * sigma4**2
        )

        x1x2, x3x4 = self._x1x2.sample((self.m,)), self._x3x4.sample((self.m,))
        self.data = torch.cat((x1x2, x3x4), dim=1)

    def __len__(self):
        return self.m

    def __getitem__(self, idx):
        return self.data[idx]

    def cond(self, x, C, M):
        _x1, _x2, _x3, _x4 = x
        if 0 not in C and 1 not in C:
            x1x2 = self._x1x2.sample((M,))
        if 0 in C and 1 in C:
            x1x2 = torch.tensor([_x1, _x2]).repeat(M, 1)
        if 0 in C and 1 not in C:
            x1 = _x1.repeat((M,))
            x2 = self._cond2(_x1).sample((M,))
            x1x2 = torch.stack((x1, x2), dim=1)
        if 0 not in C and 1 in C:
            x1 = self._cond1(_x2).sample((M,))
            x2 = _x2.repeat(M)
            x1x2 = torch.stack((x1, x2), dim=1)

        if 2 not in C and 3 not in C:
            x3x4 = self._x3x4.sample((M,))
        if 2 in C and 3 in C:
            x3x4 = torch.tensor([_x3, _x4]).repeat(M, 1)
        if 2 in C and 3 not in C:
            x3 = _x3.repeat((M,))
            x4 = self._cond4(_x3).sample((M,))
            x3x4 = torch.stack((x3, x4), dim=1)
        if 2 not in C and 3 in C:
            x3 = self._cond3(_x4).sample((M,))
            x4 = _x4.repeat(M)
            x3x4 = torch.stack((x3, x4), dim=1)

        x = torch.cat((x1x2, x3x4), dim=1)
        return x
