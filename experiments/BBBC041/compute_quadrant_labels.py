import os
import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

root_dir = "../../"
sys.path.append(root_dir)

from dataset import BBBC041Dataset

experiment_dir = os.path.join(root_dir, "experiments/BBBC041")
data_dir = os.path.join(experiment_dir, "data")
trophozoite_dir = os.path.join(data_dir, "trophozoite")

df_train = pd.read_json(os.path.join(data_dir, "training.json"))
df_test = pd.read_json(os.path.join(data_dir, "test_cropped.json"))
frame = [df_train, df_test]
df = pd.concat(frame, ignore_index=True)
image_name = []
for _, row in df.iterrows():
    image_name.append(os.path.basename(row["image"]["pathname"]).split(".")[0])
df["image_name"] = image_name
df.set_index("image_name", inplace=True)

dataset = BBBC041Dataset(os.path.join(trophozoite_dir, "val"))
image_names = [os.path.basename(x[0]).split(".")[0] for x in dataset.samples]

h, w = 1200, 1600
quadrant_df = []
flipped = []
for image_name in tqdm(image_names):
    label = 0
    ground_truth_quadrant = torch.zeros(4)
    objects = df.at[image_name, "objects"]
    for o in objects:
        category = o["category"]
        if category == "trophozoite":
            label = 1
            ground_truth = torch.zeros((1200, 1600))
            bbox = o["bounding_box"]
            ul_r = bbox["minimum"]["r"]
            ul_c = bbox["minimum"]["c"]
            br_r = bbox["maximum"]["r"]
            br_c = bbox["maximum"]["c"]
            ground_truth[ul_r : br_r + 1, ul_c : br_c + 1] = 1
            A = torch.sum(ground_truth.flatten())
            ground_truth = (
                ground_truth.unfold(0, h // 2, h // 2)
                .unfold(1, w // 2, w // 2)
                .flatten(start_dim=2)
                .sum(dim=2)
                .flatten()
            ) / A
            ground_truth_quadrant[ground_truth >= 0.80] = 1
    if torch.sum(ground_truth_quadrant) == 0 and label == 1:
        flipped.append(image_name)
    quadrant_df.append(
        {
            "image_name": image_name,
            "quadrant_labels": ground_truth_quadrant.long().tolist(),
        }
    )
print(f"Flipped {len(flipped)} labels")
np.save(os.path.join(data_dir, "flipped_images.npy"), flipped)

quadrant_df = pd.DataFrame(quadrant_df)
quadrant_df.set_index("image_name", inplace=True)
quadrant_df.to_pickle(os.path.join(data_dir, "quadrant_labels"))
quadrant_df.to_csv(os.path.join(data_dir, "quadrant_labels.csv"))
