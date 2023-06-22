import os
import uuid
import torch
import torch.nn as nn
import torchvision.transforms as t
import wandb
from torch.utils.data import DataLoader
from dataset import get_dataset
from model import BinaryClassifier
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_dir = os.path.join("checkpoints")
os.makedirs(model_dir, exist_ok=True)

augmentation = t.Compose(
    [t.RandomVerticalFlip(), t.RandomHorizontalFlip(), t.RandomRotation(90)]
)

ops = ["train", "val"]
datasets = {
    op: get_dataset(op, augmentation=augmentation if op == "train" else None)
    for op in ops
}
dataloaders = {
    op: DataLoader(d, batch_size=16, shuffle=op == "train", num_workers=4)
    for op, d in datasets.items()
}

model = BinaryClassifier()
model = model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

run_id = uuid.uuid4().hex[:4]
wandb.init(project="CelebAMask-HQ", entity="jacopoteneggi", name=f"classifier-{run_id}")

num_epochs = 25
best_model_accuracy = 0
for _ in range(num_epochs):
    for op in ops:
        if op == "train":
            torch.set_grad_enabled(True)
            model.train()
        else:
            torch.set_grad_enabled(False)
            model.eval()

        running_loss = 0.0
        running_accuracy = 0

        for i, data in enumerate(tqdm(dataloaders[op])):
            x, attr = data

            attr_name = "Smiling"
            attr_idx = datasets[op].attr_names.index(attr_name)
            target = attr[:, attr_idx].float().unsqueeze(1)

            x = x.to(device)
            target = target.to(device)

            output = model(x)
            prediction = (output >= 0.5).float()
            loss = criterion(output, target)

            running_loss += loss.item()
            running_accuracy += (prediction == target).sum().item() / target.size(0)

            if op == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                log_step = 20
                if (i + 1) % log_step == 0:
                    wandb.log(
                        {
                            f"train_loss": running_loss / log_step,
                            f"train_accuracy": running_accuracy / log_step,
                        }
                    )
                    running_loss = 0.0
                    running_accuracy = 0

        if op == "val":
            val_loss = running_loss / (i + 1)
            val_accuracy = running_accuracy / (i + 1)

            if val_accuracy > best_model_accuracy:
                best_model_accuracy = val_accuracy
                torch.save(model.state_dict(), os.path.join(model_dir, "classifier.pt"))

            wandb.log(
                {
                    f"val_loss": running_loss / (i + 1),
                    f"val_accuracy": running_accuracy / (i + 1),
                }
            )

    scheduler.step()
