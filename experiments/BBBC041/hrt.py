import os
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "8"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = "data"
trophozoite_dir = os.path.join(data_dir, "trophozoite")
explanation_dir = os.path.join("explanations")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)
model.load_state_dict(
    torch.load(os.path.join("pretrained_model", "model.pt"), map_location=device)
)
model.eval()
torch.set_grad_enabled(False)

batch_size = 4
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
dataset = ImageFolder(os.path.join(trophozoite_dir, "val"), transform)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

correct = 0
for i, data in enumerate(tqdm(dataloader)):
    input, label = data

    input = input.to(device)
    label = label.to(device)

    output = model(input)
    prediction = output.argmax(dim=1)

    correct += torch.sum(prediction == label)
t = 1 - correct / len(dataset)
print(f"Test statistic (1 - accuracy): {100*t:.2f}%")

ref = torch.load(os.path.join(explanation_dir, "reference.pt"), map_location=device)
h, w = ref.size(1), ref.size(2)
center_row, center_column = h // 2, w // 2
for j in range(4):
    print(f"Testing Y _||_ X_{j} | X_-{j}")
    row = j // 2
    col = j % 2
    if row == 0:
        start_row = 0
        end_row = center_row
    else:
        start_row = center_row
        end_row = w
    if col == 0:
        start_column = 0
        end_column = center_column
    else:
        start_column = center_column
        end_column = w

    correct = 0
    for i, data in enumerate(tqdm(dataloader)):
        input, label = data

        input = input.to(device)
        label = label.to(device)

        input[:, :, start_row : end_row + 1, start_column : end_column + 1] = ref[
            :, start_row : end_row + 1, start_column : end_column + 1
        ]
        output = model(input)
        prediction = output.argmax(dim=1)

        correct += torch.sum(prediction == label)
    t_tilde = 1 - correct / len(dataset)
    print(f"Null statistic (1 - accuracy): {100*t_tilde:.2f}%, p = {int(t >= t_tilde)}")
