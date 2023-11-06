import argparse
import os
import pickle

import torch
import torch.nn as nn
from bbbc041_utils import get_C, get_J
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = os.path.join("data")
model_dir = os.path.join("pretrained_model")

demo_image = "2f6224be-50d0-4e85-94ef-88315df561b6"
demo_image_path = os.path.join("demo", f"{demo_image}.png")

image = Image.open(demo_image_path)
w, h = image.size
gamma = 16

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)

model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(
    torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
)
model = model.to(device)
model.eval()
x = torch.randn(1, 3, h, w, device=device)
model(x)
torch.cuda.empty_cache()

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
)
image = Image.open(demo_image_path)
image_t = transform(image).to(device)

ref = torch.load(os.path.join("demo", "reference.pt"), map_location=device)

J = get_J(gamma, h, w)

F = {}
for _C in tqdm(get_C(gamma)):
    input = torch.clone(ref)
    for j in _C:
        input[:, J[j][0][0] : J[j][1][0], J[j][0][1] : J[j][1][1]] = image_t[
            :, J[j][0][0] : J[j][1][0], J[j][0][1] : J[j][1][1]
        ]
    output = model(input.unsqueeze(0))
    F[tuple(sorted(_C))] = torch.softmax(output, dim=1)[0, 1].item()

with open(os.path.join("demo", f"F_{gamma}.pkl"), "wb") as f:
    pickle.dump(F, f)
