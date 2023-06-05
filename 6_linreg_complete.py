import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_regression

epochs = 500
lr = 0.1
loss_stop_threshold = 0.01
n_samples = 100

x_np, y_np = make_regression(n_samples=n_samples, n_features=1, noise=5)
x, y = (
    torch.from_numpy(x_np.astype(np.float32)),
    torch.from_numpy(y_np.astype(np.float32)),
)

x_norm = F.normalize(x, dim=0)
y = y.view((-1, 1))

n_instances, n_feats = x.shape
_, output_size = y.shape


class LinReg(nn.Module):
    def __init__(self, input_dims, output_dims, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.lin = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        return self.lin(x)


m = LinReg(n_feats, output_size)
l = nn.MSELoss()
opt = torch.optim.SGD(m.parameters(), lr=lr)

for ep in range(epochs):
    y_pred = m(x)
    loss_i = l(y, y_pred)

    if loss_i < loss_stop_threshold:
        print(f"stopping at epoch {ep}")
        break

    loss_i.backward()

    opt.step()
    opt.zero_grad()

    w, b = m.parameters()
    if ep % 5 == 0:
        print(f"loss_i: {loss_i}")
        print(f"epoch #{ep} | loss: {loss_i} | w: {w[0][0]} | b: {b[0]} ")
        print()

preds = m(x).detach().numpy()

import matplotlib.pyplot as plt

plt.figure()
plt.plot(x_np, y_np, 'ro')
plt.plot(x_np, preds.reshape(-1), 'b')
plt.show()
