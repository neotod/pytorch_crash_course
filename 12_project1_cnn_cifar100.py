import torch
import os
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

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


image_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=cifar100_mean, std=cifar100_std)]
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


class Cifar100Clf(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
        )
        self.init_weights(self.conv_layer1)

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
        )
        self.init_weights(self.conv_layer2)

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
        )
        self.init_weights(self.conv_layer3)

        self.lin_layer1 = nn.Sequential(
            nn.Linear(64, 128), nn.LeakyReLU(), nn.Dropout(p=0.7)
        )
        self.init_weights(self.lin_layer1)

        self.lin_layer2 = nn.Sequential(
            nn.Linear(128, 256), nn.LeakyReLU(), nn.Dropout(p=0.1)
        )
        self.init_weights(self.lin_layer2)

        self.output_layer = nn.Linear(256, 100)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)

        x = x.view(-1, 64)

        x = self.lin_layer1(x)
        x = self.lin_layer2(x)
        x = self.output_layer(x)

        return x


m = Cifar100Clf().to(device)
opt = torch.optim.Adam(m.parameters(), lr=learning_rate)
loss_ = nn.CrossEntropyLoss()

for ep in range(epochs):
    train_loss = 0
    for (x, y) in train_loader:
        x = x.to(device)
        y = y.to(device)

        y_pred = m(x)

        loss_i = loss_(y_pred, y)
        train_loss += loss_i

        opt.zero_grad()
        loss_i.backward()
        opt.step()

    # validation
    val_loss = 0
    for (x, y) in val_loader:
        x = x.to(device)
        y = y.to(device)

        y_pred = m(x)

        loss_i = loss_(y_pred, y)
        val_loss += loss_i

    print(
        f"ep: {ep} | loss: {train_loss / batch_size} | val_loss: {val_loss / batch_size}"
    )

# accuracy
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for (x, y) in test_loader:
        x = x.to(device)
        y = y.to(device)

        y_pred = m(x)

        _, predicted = torch.max(y_pred, 1)
        n_samples += y.size(0)
        n_correct += (predicted == y).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the network: {acc} %")

