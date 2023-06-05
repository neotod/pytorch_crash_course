import time
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 784  # 28 * 28
n_classes = 10
learning_rate = 10
epochs = 100


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim

        self._hidden_sizes = [512, 128]
        self.hidden_layers = nn.ModuleList()

        prev_size = input_dim
        for size in self._hidden_sizes:
            x = nn.Linear(prev_size, size)
            self.hidden_layers.append(x)

            prev_size = size

        self.output_layer = nn.Linear(prev_size, self.output_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = nn.ReLU()(x)

        return self.output_layer(x)

    def params_num(self):
        num = 0
        for parameter in self.parameters():
            num += len(parameter)

        return num

m = FeedForwardNN(15, 10)
lss = nn.CrossEntropyLoss()

train_dataset = torchvision.datasets.MNIST(
    "./data", train=True, transform=transforms.ToTensor()
)
test_dataset = torchvision.datasets.MNIST(
    "./data", train=False, transform=transforms.ToTensor()
)


batch_size = 16
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


m = FeedForwardNN(input_size, n_classes).to(device)
loss_ = nn.CrossEntropyLoss()
opt = torch.optim.Adam(m.parameters(), lr=learning_rate)

print("the model:")
print(m)
print(f"model got {m.params_num()} parameters")


print(f"running the code on device is: {device}")
for ep in range(epochs):
    t1 = time.perf_counter()
    for x, y in train_loader:
        x = x.reshape((-1, 1, 28 * 28)).to(device)
        y = y.to(device)

        y_pred = m(x)
        y_pred = y_pred.reshape((batch_size, n_classes))

        loss_i = loss_(y_pred, y)

        opt.zero_grad()
        loss_i.backward()
        opt.step()

    t2 = time.perf_counter()

    print(f"epoch #{ep} | loss: {loss_i} | time: {t2 - t1}")


with torch.no_grad():
    y_pred = m()

    n_correct = 0

    for (x, y) in test_dataset:
        y_pred = m(x)

        print(y_pred)
        
