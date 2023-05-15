import os
import torch
import torchvision.transforms as t
from torchvision.datasets import CelebA
from PIL import Image
from torchvision.datasets.utils import verify_str_arg

MU = torch.tensor([0.5174, 0.4169, 0.3636])
STD = torch.tensor([0.3028, 0.2743, 0.2692])


def get_dataset(
    split, mu=MU, std=STD, augmentation=None, return_feature=False, return_idx=False
):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    root = os.path.join(data_dir, "CelebAMask-HQ")

    img_size = 256
    transform = t.Compose(
        [
            t.ToTensor(),
            t.Resize(img_size),
            t.Normalize(mu, std),
        ]
    )
    if augmentation is not None:
        transform = t.Compose([transform, augmentation])

    feature_transform = t.Resize(img_size)

    dataset = CelebAMaskHQ(
        root,
        split=split,
        return_feature=return_feature,
        transform=transform,
        feature_transform=feature_transform,
        return_idx=return_idx,
    )
    return dataset


class CelebAMaskHQ(CelebA):
    base_folder = "celebamaskhq"

    def __init__(
        self,
        root,
        split=None,
        return_feature=False,
        transform=None,
        feature_transform=None,
        return_idx=False,
    ):
        self.root = root
        self.split = split
        self.return_feature = return_feature
        self.transform = transform
        self.feature_transform = feature_transform
        self.return_idx = return_idx

        split_map = {"train": 0, "val": 1, "all": None}
        _split = split_map[
            verify_str_arg(split.lower(), "split", ("train", "val", "all"))
        ]
        splits = self._load_csv("CelebAMask-HQ-partition.txt")
        complete = self._load_csv("CelebAMask-HQ-complete.txt")
        attr = self._load_csv("CelebAMask-HQ-attribute-anno.txt", header=1)

        if return_feature:
            mask = (
                (complete.data == 1).squeeze()
                if _split is None
                else torch.logical_and(
                    splits.data == _split, complete.data == 1
                ).squeeze()
            )
        else:
            mask = slice(None) if _split is None else (splits.data == _split).squeeze()

        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [
                splits.index[i] for i in torch.squeeze(torch.nonzero(mask))
            ]
        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        self.attr_names = attr.header

    def __getitem__(self, index):
        filename = self.filename[index]

        x = Image.open(
            os.path.join(self.root, self.base_folder, "CelebA-HQ-img", filename)
        )

        if self.transform is not None:
            x = self.transform(x)

        attr = self.attr[index, :]
        if self.return_feature:
            i = int(filename.split(".")[0])
            m = i // 2000
            feature = torch.load(
                os.path.join(
                    self.root,
                    self.base_folder,
                    "CelebAMask-HQ-feature",
                    str(m),
                    f"{i:05d}.pt",
                )
            )

            if self.feature_transform is not None:
                feature = self.feature_transform(feature)

            if self.return_idx:
                return index, x, attr, feature
            else:
                return x, attr, feature
        else:
            if self.return_idx:
                return index, x, attr
            else:
                return x, attr
