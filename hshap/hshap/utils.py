import numpy as np
import torch
from torch import Tensor

factorial = np.math.factorial


def hshap_features(gamma: int) -> np.ndarray:
    return np.expand_dims(np.eye(gamma, dtype=np.bool_), axis=1)


def w(c: int, gamma: int) -> int:
    return factorial(c) * factorial(gamma - c - 1) / factorial(gamma)


def shapley_matrix(gamma: int, device: torch.device) -> Tensor:
    if gamma == 2:
        W = torch.tensor(
            [
                [-w(0, gamma), w(0, gamma), -w(1, gamma), w(1, gamma)],
                [-w(0, gamma), -w(1, gamma), w(0, gamma), w(1, gamma)],
            ],
            device=device,
        )
    elif gamma == 4:
        # construct matrix as copies of first row
        W = torch.tensor(
            [
                [-w(0, gamma), w(0, gamma)]
                + 3 * [-w(1, gamma)]
                + 3 * [w(1, gamma)]
                + 3 * [-w(2, gamma)]
                + 3 * [w(2, gamma)]
                + [-w(3, gamma), w(3, gamma)]
            ],
            device=device,
        ).repeat(gamma, 1)
        # update second row
        W[1, 1] = -w(1, gamma)
        W[1, 2] = w(0, gamma)
        W[1, 6] = -w(2, gamma)
        W[1, 7] = -w(2, gamma)
        W[1, 8] = w(1, gamma)
        W[1, 9] = w(1, gamma)
        W[1, -3] = -w(3, gamma)
        W[1, -2] = w(2, gamma)
        # update third row
        W[2, 1] = -w(1, gamma)
        W[2, 3] = w(0, gamma)
        W[2, 5] = -w(2, gamma)
        W[2, 7] = -w(2, gamma)
        W[2, 8] = w(1, gamma)
        W[2, 10] = w(1, gamma)
        W[2, -4] = -w(3, gamma)
        W[2, -2] = w(2, gamma)
        # update fourth row
        W[3, 1] = -w(1, gamma)
        W[3, 4] = w(0, gamma)
        W[3, 5] = -w(2, gamma)
        W[3, 6] = -w(2, gamma)
        W[3, 9] = w(1, gamma)
        W[3, 10] = w(1, gamma)
        W[3, -5] = -w(3, gamma)
        W[3, -2] = w(2, gamma)
    else:
        raise NotImplementedError("Only implemented for gamma equals to 2 or 4")

    return W.transpose(0, 1)


def mask_features_(
    feature_mask: Tensor,
    root_coords: np.ndarray,
) -> None:
    center = np.mean(root_coords, axis=0, dtype=np.uint16)

    feature_mask[
        1, :, root_coords[0, 0] : center[0], root_coords[0, 1] : center[1]
    ].fill_(True)
    feature_mask[
        2, :, root_coords[0, 0] : center[0], center[1] : root_coords[1, 1]
    ].fill_(True)
    feature_mask[
        3, :, center[0] : root_coords[1, 0], root_coords[0, 1] : center[1]
    ].fill_(True)
    feature_mask[
        4, :, center[0] : root_coords[1, 0], center[1] : root_coords[1, 1]
    ].fill_(True)


def mask_input_(
    input: Tensor,
    path: np.ndarray,
    background: Tensor,
    root_coords: np.ndarray,
) -> None:
    if not np.all(path):
        center = np.mean(root_coords, axis=0, dtype=np.uint16)

        if not path[0]:
            input[
                :, root_coords[0, 0] : center[0], root_coords[0, 1] : center[1]
            ] = background[
                :, root_coords[0, 0] : center[0], root_coords[0, 1] : center[1]
            ]
        if not path[1]:
            input[
                :, root_coords[0, 0] : center[0], center[1] : root_coords[1, 1]
            ] = background[
                :, root_coords[0, 0] : center[0], center[1] : root_coords[1, 1]
            ]
        if not path[2]:
            input[
                :, center[0] : root_coords[1, 0], root_coords[0, 1] : center[1]
            ] = background[
                :, center[0] : root_coords[1, 0], root_coords[0, 1] : center[1]
            ]
        if not path[3]:
            input[
                :, center[0] : root_coords[1, 0], center[1] : root_coords[1, 1]
            ] = background[
                :, center[0] : root_coords[1, 0], center[1] : root_coords[1, 1]
            ]

        feature_id = np.nonzero(path)[0][0]
        feature_row, feature_column = feature_id // 2, feature_id % 2
        root_coords[0, 0] = center[0] if feature_row == 1 else root_coords[0, 0]
        root_coords[0, 1] = center[1] if feature_column == 1 else root_coords[0, 1]
        root_coords[1, 0] = center[0] if (1 - feature_row) == 1 else root_coords[1, 0]
        root_coords[1, 1] = (
            center[1] if (1 - feature_column) == 1 else root_coords[1, 1]
        )


def mask_map_(
    map: Tensor,
    path: np.ndarray,
    score: float,
    root_coords: np.ndarray,
):
    center = np.mean(root_coords, axis=0, dtype=np.uint16)

    if path[0]:
        map[:, root_coords[0, 0] : center[0], root_coords[0, 1] : center[1]].fill_(
            score
        )
    elif path[1]:
        map[:, root_coords[0, 0] : center[0], center[1] : root_coords[1, 1]].fill_(
            score
        )
    elif path[2]:
        map[:, center[0] : root_coords[1, 0], root_coords[0, 1] : center[1]].fill_(
            score
        )
    elif path[3]:
        map[:, center[0] : root_coords[1, 0], center[1] : root_coords[1, 1]].fill_(
            score
        )


def shaplit_sets(gamma: int):
    if gamma != 4:
        raise NotImplementedError("Only implemented for gamma equals to 4")
    else:
        C = [
            [{}, {2}, {3}, {4}, {2, 3}, {2, 4}, {3, 4}, {2, 3, 4}],
            [{}, {1}, {3}, {4}, {1, 3}, {1, 4}, {3, 4}, {1, 3, 4}],
            [{}, {1}, {2}, {4}, {1, 2}, {1, 4}, {2, 4}, {1, 2, 4}],
            [{}, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}],
        ]

    return C


def shaplit_matrix(gamma: int, device: torch.device) -> Tensor:
    if gamma != 4:
        raise NotImplementedError("Only implemented for gamma equals to 4")
    else:
        M = torch.empty((gamma, 2 ** (gamma - 1), 2), device=device).long()
        # first feature
        M[0, 0, 0] = 1
        M[0, 0, 1] = 0
        M[0, 1, 0] = 5
        M[0, 1, 1] = 2
        M[0, 2, 0] = 6
        M[0, 2, 1] = 3
        M[0, 3, 0] = 7
        M[0, 3, 1] = 4
        M[0, 4, 0] = 11
        M[0, 4, 1] = 8
        M[0, 5, 0] = 12
        M[0, 5, 1] = 9
        M[0, 6, 0] = 13
        M[0, 6, 1] = 10
        M[0, 7, 0] = 15
        M[0, 7, 1] = 14
        # second feature
        M[1, 0, 0] = 2
        M[1, 0, 1] = 0
        M[1, 1, 0] = 5
        M[1, 1, 1] = 1
        M[1, 2, 0] = 8
        M[1, 2, 1] = 3
        M[1, 3, 0] = 9
        M[1, 3, 1] = 4
        M[1, 4, 0] = 11
        M[1, 4, 1] = 6
        M[1, 5, 0] = 12
        M[1, 5, 1] = 7
        M[1, 6, 0] = 14
        M[1, 6, 1] = 10
        M[1, 7, 0] = 15
        M[1, 7, 1] = 13
        # third feature
        M[2, 0, 0] = 3
        M[2, 0, 1] = 0
        M[2, 1, 0] = 6
        M[2, 1, 1] = 1
        M[2, 2, 0] = 8
        M[2, 2, 1] = 2
        M[2, 3, 0] = 10
        M[2, 3, 1] = 4
        M[2, 4, 0] = 11
        M[2, 4, 1] = 5
        M[2, 5, 0] = 13
        M[2, 5, 1] = 7
        M[2, 6, 0] = 14
        M[2, 6, 1] = 9
        M[2, 7, 0] = 15
        M[2, 7, 1] = 12
        # fourth feature
        M[3, 0, 0] = 4
        M[3, 0, 1] = 0
        M[3, 1, 0] = 7
        M[3, 1, 1] = 1
        M[3, 2, 0] = 9
        M[3, 2, 1] = 2
        M[3, 3, 0] = 10
        M[3, 3, 1] = 3
        M[3, 4, 0] = 12
        M[3, 4, 1] = 5
        M[3, 5, 0] = 13
        M[3, 5, 1] = 6
        M[3, 6, 0] = 14
        M[3, 6, 1] = 8
        M[3, 7, 0] = 15
        M[3, 7, 1] = 11

        return M
