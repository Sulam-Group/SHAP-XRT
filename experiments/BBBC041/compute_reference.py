import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

data_dir = "data"
trophozoite_dir = os.path.join(data_dir, "trophozoite")

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

train_dataset = ImageFolder(os.path.join(trophozoite_dir, "train"), transform)
dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=100, shuffle=True, num_workers=0
)
_iter = iter(dataloader)
X, _ = next(_iter)
ref = X.detach().mean(0)

torch.save(ref, os.path.join("explanations", "reference.pt"))
