from torchinfo import summary
from typing import Any
import torch
import os
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 100
batch_size = 16
learning_rate = 0.1
val_size = 0.2
model_saving_path = "model.pth"


download_ds = False
if not os.path.exists(os.path.join("./data", "cifar-100-python")):
    download_ds = True

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)


class PretainedModelParamsPreProc:
    def __init__(self, preproc_func) -> None:
        self.preproc_func = preproc_func

    def __call__(self, sample) -> Any:
        x, y = sample
        x = self.preproc_func(x)

        return x, y


image_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
        PretainedModelParamsPreProc(ResNet50_Weights.transforms),
    ]
)

train_val_ds = torchvision.datasets.CIFAR100(
    "./data", download=download_ds, transform=image_transform, train=True
)
test_ds = torchvision.datasets.CIFAR100(
    "./data", download=False, transform=image_transform, train=False
)

train_size = int(len(train_val_ds) * (1 - val_size))
val_size = int(len(train_val_ds) * (val_size))

train_ds, val_ds = torch.utils.data.random_split(train_val_ds, [train_size, val_size])

train_loader = DataLoader(dataset=train_ds, batch_size=batch_size)
test_loader = DataLoader(dataset=test_ds, batch_size=1)
val_loader = DataLoader(dataset=val_ds, batch_size=batch_size)


class Cifar100ClfPretained(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._resnet50 = torchvision.models.resnet50(weights=ResNet50_Weights)

        for param in self._resnet50.parameters():
            param.requires_grad = False

        modules = list(self._resnet50.children())[
            :-1
        ]  # all the layers except the last fully connected layer
        resnet50_fclayer_in_feats = self._resnet50.fc.in_features

        self.feat_extractor = nn.Sequential(*modules)

        self.fc1 = nn.Sequential(
            nn.Linear(resnet50_fclayer_in_feats, 512), nn.ReLU(), nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(nn.Linear(512, 100), nn.ReLU(), nn.Dropout(p=0.5))

    def forward(self, x):
        feats = self.feat_extractor(x)

        x = self.fc1(feats)
        x = self.fc2(x)

        return x


m = Cifar100ClfPretained().to(device)

print(
    summary(
        model=m,
        input_size=(
            32,
            3,
            224,
            224,
        ),  # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )
)

# loss_ = nn.CrossEntropyLoss()
# opt = torch.optim.Adam(m.parameters(), lr=learning_rate)
# step_lr_sched = torch.optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.1)

# for ep in range(epochs):
#     ep_train_loss_sum = 0
#     for (x, y) in train_loader:
#         x = x.to(device)
#         y = y.to(device)

#         y_pred = m(x)

#         loss_i = loss_(y_pred, y)
#         ep_train_loss_sum += loss_i

#     ep_val_loss_sum = 0
#     for (x, y) in train_loader:
#         x = x.to(device)
#         y = y.to(device)

#         y_pred = m(x)

#         loss_i = loss_(y_pred, y)
#         ep_val_loss_sum += loss_i

#     print(
#         f"epoch: {ep} | train_loss: {ep_train_loss_sum / batch_size} | val_loss: {ep_val_loss_sum / batch_size}"
#     )
