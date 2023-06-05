import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 100
lr = 0.1
batch_size = 16

train_ds = CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=True)
test_ds = CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=1, shuffle=False)

class Cifar10Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.max_pool1 = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(6,16, 5)
        self.max_pool2 = nn.MaxPool2d((2,2))

        self.fc1 = nn.Linear(16*5*5 / 2, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.max_pool1(F.relu(self.conv1(x)))
        x = self.max_pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

input_dim, output_dim = (32, 32), 10
m = Cifar10Classifier(input_dim, output_dim)
loss_ = nn.CrossEntropyLoss()
opt = torch.optim.Adam(m.parameters(),  lr=lr)

for ep in range(epochs):
    epoch_loss = 0
    for (x, y) in train_loader:
        x = x.to(device)
        y = y.to(device)

        y_pred = m(x)

        loss_i = loss_(y_pred, y)
        opt.zero_grad()
        loss_i.backward()
        opt.step()

        epoch_loss += loss_i

    print(f'epoch {ep} | loss: {epoch_loss / batch_size}')
        

m.conv1.weight.grad