import os
import sys

import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

root_dir = "../../"
sys.path.append(root_dir)

from dataset import Crosses
from utils import alpha, batch_size, d, models, r, s, shapxrt_power, train

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# `df` is the output DataFrame
# `sigma` is the noise level
# `K` is the number of null statistics to compute
df = []
sigma = 1 / d
K = 1000
# for each model
for model in models:
    model_name, module, optimizer, lr = model
    # for each number of samples between 1e01 and 1e04
    for k in tqdm(10 ** np.linspace(1, 4, 10)):
        k = int(k)
        m = batch_size * k
        # for 10 independent repetitions
        for i in range(10):
            # initialize training and test datasets
            train_dataset = Crosses(m, r, s, d, sigma)
            test_dataset = Crosses(10 * batch_size, r, s, d, sigma)

            # initialize training dataloader
            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, num_workers=4
            )

            # initialize the model
            if model_name == "CNN":
                net = module(d)
            else:
                net = module(d, r, s)

            # initialize the optimizer
            optim = optimizer(net.parameters(), lr=lr)

            # train the model
            net, acc = train(train_dataloader, test_dataset, net, optim, nn.BCELoss())

            # estimate the power
            beta = shapxrt_power(net, test_dataset, alpha, K)
            df.append({"model_name": model_name, "m": m, "beta": beta})

df = pd.DataFrame(df)
df.to_csv(os.path.join(data_dir, "power_m.csv"), index=False)
