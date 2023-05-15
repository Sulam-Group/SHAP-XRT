import os
import torch
import torchvision.transforms.functional as tf
import csv
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm

data_dir = "data"
dataset_dir = os.path.join(data_dir, "CelebAMask-HQ/celebamaskhq")
segmentation_dir = os.path.join(dataset_dir, "CelebAMask-HQ-mask-anno")
feature_dir = os.path.join(dataset_dir, "CelebAMask-HQ-feature")

_mask_name = lambda x: "_".join((x.split(".")[0]).split("_")[1:])
_mask_layer = {
    "cloth": 0,
    "neck": 0,
    "neck_l": 0,
    "skin": 0,
    "l_ear": 0,
    "r_ear": 0,
    "ear_r": 0,
    "mouth": 1,
    "l_lip": 1,
    "u_lip": 1,
    "nose": 2,
    "l_brow": 2,
    "r_brow": 2,
    "r_eye": 2,
    "l_eye": 2,
    "eye_g": 2,
    "hair": 3,
    "hat": 3,
}
d = max(_mask_layer.values()) + 1


def _mask_to_feature(i):
    m = i // 2000
    segmentation_group_dir = os.path.join(segmentation_dir, str(m))
    filenames = list(
        filter(lambda x: x.startswith(f"{i:05d}"), os.listdir(segmentation_group_dir))
    )
    mask_name = [_mask_name(f) for f in filenames]
    mask_layer = torch.tensor([_mask_layer[m] for m in mask_name]).long()
    if set(mask_layer.tolist()) != set(range(d)):
        return [i, 0]

    mask = torch.stack(
        [
            torch.mean(
                tf.to_tensor(Image.open(os.path.join(segmentation_group_dir, f))), dim=0
            ).long()
            for f in filenames
        ]
    )
    layer = torch.stack(
        [(j + 1) * torch.amax(mask[mask_layer == j], dim=0) for j in range(d)]
    )
    feature = torch.amax(layer, dim=0)
    feature = feature.unsqueeze(0)

    feature_group_id = os.path.join(feature_dir, str(m))
    os.makedirs(feature_group_id, exist_ok=True)
    torch.save(feature, os.path.join(feature_group_id, f"{i:05d}.pt"))
    return [i, 1]


n = int(30e03)
complete = Parallel(n_jobs=32)(delayed(_mask_to_feature)(i) for i in tqdm(range(n)))

with open(os.path.join(dataset_dir, "CelebAMask-HQ-complete.txt"), "w") as f:
    writer = csv.writer(f, delimiter=" ")
    writer.writerows(complete)
