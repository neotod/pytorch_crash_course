import torch, numpy as np

epochs = 100
lr = 0.1
loss_stop_threshold = 0.01

# x, y = make_regression(n_samples=100, n_features=1)

x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = abs(torch.randn(1))


def forward(x):
    return w * x


def loss(y_true, y_pred):
    # mse loss

    # print(f"(y_true - y_pred) ** 2 - {(y_true - y_pred) ** 2}")

    return ((y_pred - y_true) ** 2).mean()


def gradient_w(y_true, y_pred, x):
    # print(f"gradient_w - {2 * (y_true - y_pred)}")

    return (2 * (y_pred - y_true) * x).mean()


for ep in range(epochs):
    i = torch.randint(low=0, high=x.size()[0], size=(1,))[0]
    y_pred = forward(x[i])
    y_true = y[i]

    loss_i = loss(y_true, y_pred)

    if loss_i < loss_stop_threshold:
        print(f"stopping at epoch {ep}")
        break

    w_grad = gradient_w(y_true, y_pred, x[i])
    w -= lr * w_grad

    print(f"epoch #{ep} | loss: {loss_i} | w: {w}")
    print()

