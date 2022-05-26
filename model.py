import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimpleCNN(nn.Module):
    """
    A class that implements a simple CNN with one filter as in Eq. (47) of the paper.
    """

    def __init__(self, d: int) -> None:
        """
        Initialize the CNN.

        Parameters:
        -----------
        d: int
            the dimension (in pixels) of each patch.
        """
        super(SimpleCNN, self).__init__()
        self.w = nn.Conv2d(1, 1, kernel_size=(d, d), stride=d, bias=True)
        nn.init.kaiming_normal_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the CNN.
        """
        x = self.w(x)
        x = x.flatten(start_dim=1).sum(dim=1)
        x = torch.sigmoid(x)
        return x


class SimpleFCN(nn.Module):
    """
    A class that implements a two-layer FCN with ReLU activation as in Eq. (48) of the paper.
    """

    def __init__(self, d: int, r: int, s: int) -> None:
        """
        Initialize the FCN.

        Parameters:
        -----------
        d: int
            the dimension (in pixels) of each patch.
        r: int
            the number of rows of patches in each sample.
        s: int
            the number of columns of patches in each sample.
        """
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(d**2 * r * s, 64)
        self.fc2 = nn.Linear(64, 1)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the FCN.
        """
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x).squeeze()
        return x


class BooleanF(nn.Module):
    """
    A class that implements the ground-truth Boolean function
    in Eq. (12) of the paper.
    """

    def __init__(self, n: int, t: float) -> None:
        """
        Initialize the function.

        Parameters:
        -----------
        n: int
            the number of features in each component.
        t: float
            the threshold of the function.
        """
        super(BooleanF, self).__init__()
        self.n = n
        self.t = t

    def forward(self, x):
        """
        Foward pass of the function.
        """
        x = x.abs() >= self.t
        x = x.unfold(1, self.n, self.n)
        x = x.any(dim=2)
        x = x.all(dim=1)
        x = x.float()
        return x
