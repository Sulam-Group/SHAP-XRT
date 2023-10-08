import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import hshap

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
)

ref = torch.load(os.path.join(explanation_dir, "reference.pt"), map_location=device)
hexp = hshap.src.Explainer(
    model=model,
    background=ref,
)
print("Initialized hshap")

for s in [800, 400]:
    true_positives = np.load(os.path.join(explanation_dir, "true_positive.npy"))
    for i, image_path in enumerate(true_positives):
        image_name = os.path.basename(image_path).split(".")[0]
        try:
            image = Image.open(image_path)
        except:
            print(f"skipped {i}")
        image_t = transform(image).to(device)
        threshold_mode = "relative"
        threshold_value = 60
        t0 = time.time()
        explanation = hexp.explain(
            image_t,
            label=1,
            s=s,
            threshold_mode=threshold_mode,
            threshold=threshold_value,
            output_threshold=0.55,
            batch_size=2,
            binary_map=True,
            return_shaplit=True,
        )
        torch.cuda.empty_cache()
        runtime = round(time.time() - t0, 6)
        print(f"{i+1}/{len(true_positives)} runtime={runtime:4f} s")
        with open(os.path.join(explanation_dir, f"{image_name}_{s}.pkl"), "wb") as f:
            pickle.dump(explanation, f)
