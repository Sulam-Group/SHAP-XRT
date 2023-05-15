import os
import csv

data_dir = "data"
dataset_dir = os.path.join(data_dir, "CelebAMask-HQ/celebamaskhq")

n = int(30e03)
split_all = []
for i in range(n):
    split_all.append([f"{i}.jpg", 0])

with open(os.path.join(dataset_dir, "CelebAMask-HQ-partition.txt"), "w") as f:
    writer = csv.writer(f, delimiter=" ")
    writer.writerows(split_all)
