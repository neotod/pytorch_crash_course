import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 784  # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 16
learning_rate = 0.001


class FeedForwardNNSimple(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

    def params_num(self):
        num = 0
        for parameter in self.parameters():
            num += len(parameter)

        return num


# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor()
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor()
)

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# examples = iter(test_loader)
# example_data, example_targets = next(examples)

# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(example_data[i][0], cmap='gray')
# plt.show()



m = FeedForwardNNSimple(input_size, hidden_size, num_classes).to(device)
loss_ = nn.CrossEntropyLoss()
opt = torch.optim.Adam(m.parameters(), lr=learning_rate)

print("the model:")
print(m)
print(f"model got {m.params_num()} parameters")


print(f"running the code on device is: {device}")
for ep in range(num_epochs):
    t1 = time.perf_counter()
    for x, y in train_loader:
        x = x.reshape((-1, 28 * 28)).to(device)
        y = y.to(device)

        y_pred = m(x)
        y_pred = y_pred.reshape((batch_size, -1))

        loss_i = loss_(y_pred, y)

        opt.zero_grad()
        loss_i.backward()
        opt.step()

    t2 = time.perf_counter()

    print(f"epoch #{ep} | loss: {loss_i} | time: {t2 - t1}")
