import torch, numpy as np
import torch.nn as nn

epochs = 100
lr = 0.1
loss_stop_threshold = 0.01

# x, y = make_regression(n_samples=100, n_features=1)

x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

print(w)


def forward(x):
    return w * x


def loss(y_true, y_pred):
    # mse loss

    # print(f"(y_true - y_pred) ** 2 - {(y_true - y_pred) ** 2}")

    return ((y_pred - y_true) ** 2).mean()


def gradient_w(y_true, y_pred, x):
    # print(f"gradient_w - {2 * (y_true - y_pred)}")

    return (2 * (y_pred - y_true) * x).mean()


l = nn.MSELoss()
opt = torch.optim.SGD([w], lr=lr)

for ep in range(epochs):
    # i = torch.randint(low=0, high=x.size()[0], size=(1,))[0]
    y_pred = forward(x)
    y_true = y

    loss_i = l(y_true, y_pred)

    print(f"loss_i: {loss_i}")

    if loss_i < loss_stop_threshold:
        print(f"stopping at epoch {ep}")
        break

    loss_i.backward()
    opt.step()
    opt.zero_grad()

    print(f"epoch #{ep} | loss: {loss_i} | w: {w}")
    print()

