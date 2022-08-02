import os
import argparse
import torch
import torch.nn as nn
import pickle
import time
import hshap
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--s", type=int, default=800)
parser.add_argument("--threshold_mode", type=str, default="absolute")
parser.add_argument("--threshold", type=int, default=0)
parser.add_argument("--logit_threshold", type=float, default=0.55)
args = parser.parse_args()

s = args.s
threshold_mode = args.threshold_mode
threshold = args.threshold
logit_threshold = args.logit_threshold

os.environ["CUDA_VISIBLE_DEVICES"] = "8"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = os.path.join("data")
trophozoite_dir = os.path.join(data_dir, "trophozoite")
model_dir = os.path.join("pretrained_model")
explanation_dir = os.path.join("explanations")

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
x = torch.randn(1, 3, 1200, 1600, device=device)
model(x)
torch.cuda.empty_cache()

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
dataset = ImageFolder(os.path.join(trophozoite_dir, "val"), transform)
image_names = [os.path.basename(x[0]).split(".")[0] for x in dataset.samples]
dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

ref = torch.load(os.path.join(explanation_dir, "reference.pt"), map_location=device)
hexp = hshap.src.Explainer(
    model=model,
    background=ref,
)
explainer_dir = os.path.join(
    explanation_dir, "hexp", str(s), f"{threshold_mode}_{threshold}"
)
os.makedirs(explainer_dir, exist_ok=True)
print("Initialized hshap")

for i, data in enumerate(dataloader):
    input, _ = data

    input = input.to(device)
    output = model(input)
    prediction = output.argmax(dim=1)

    if prediction == 1:
        image_name = image_names[i]
        t0 = time.time()
        explanation = hexp.explain(
            input,
            label=1,
            s=s,
            threshold_mode=threshold_mode,
            threshold=threshold,
            softmax_activation=True,
            logit_threshold=logit_threshold,
            batch_size=2,
            binary_map=True,
            return_shaplit=True,
        )
        torch.cuda.empty_cache()
        runtime = round(time.time() - t0, 6)
        print(f"{i+1}/{len(dataset)} runtime={runtime:4f} s")
        with open(
            os.path.join(explainer_dir, f"{image_name}_{logit_threshold}.pkl"),
            "wb",
        ) as f:
            pickle.dump(explanation, f)
