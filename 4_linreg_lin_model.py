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

lin = nn.Linear(n_feats, output_size)

print("init params")
w, b = lin.parameters()
print(f"w: {w[0][0]} | b: {b[0]} ")


l = nn.MSELoss()
opt = torch.optim.SGD(lin.parameters(), lr=lr)

for ep in range(epochs):
    y_pred = lin(x_norm)
    loss_i = l(y, y_pred)


    if loss_i < loss_stop_threshold:
        print(f"stopping at epoch {ep}")
        break

    loss_i.backward()

    # print("grads:")
    # w, b = lin.parameters()
    # print(w.grad)
    # print(b.grad)
    # print()

    opt.step()
    opt.zero_grad()

    w, b = lin.parameters()
    if ep % 50 == 0:
        print(f"loss_i: {loss_i}")
        print(f"epoch #{ep} | loss: {loss_i} | w: {w[0]} | b: {b[0]} ")
        print()

