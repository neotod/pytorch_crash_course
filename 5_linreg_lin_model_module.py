import torch
import torch.nn as nn
import torch.nn.functional as F

epochs = 500
lr = 0.1
loss_stop_threshold = 0.01

x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

x_norm = F.normalize(x, dim=0)

n_instances, n_feats = x.shape
_, output_size = y.shape


class LinReg(nn.Module):
    def __init__(self, input_dims, output_dims, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.lin = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        return self.lin(x)


m = LinReg(n_feats, output_size)

print("init params")
w, b = m.parameters()
print(f"w: {w[0][0]} | b: {b[0]} ")

l = nn.MSELoss()
opt = torch.optim.SGD(m.parameters(), lr=lr)

for ep in range(epochs):
    y_pred = m(x_norm)
    loss_i = l(y, y_pred)

    if loss_i < loss_stop_threshold:
        print(f"stopping at epoch {ep}")
        break

    loss_i.backward()

    opt.step()
    opt.zero_grad()

    w, b = m.parameters()
    if ep % 50 == 0:
        print(f"loss_i: {loss_i}")
        print(f"epoch #{ep} | loss: {loss_i} | w: {w[0]} | b: {b[0]} ")
        print()

