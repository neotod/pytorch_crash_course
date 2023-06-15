import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

epochs = 5000
lr = 0.01
train_stop_loss_th = 0.1

x_np, y_np = load_breast_cancer(return_X_y=True)

y_np = y_np.reshape((-1, 1))

x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(
    x_np, y_np, test_size=0.2
)

sc = StandardScaler()
x_train_np = sc.fit_transform(x_train_np)
x_test_np = sc.transform(x_test_np)


x_train = torch.from_numpy(x_train_np.astype(np.float32))
y_train = torch.from_numpy(y_train_np.reshape((-1, 1)).astype(np.float32))

x_test = torch.from_numpy(x_test_np.astype(np.float32))
y_test = torch.from_numpy(y_test_np.reshape((-1, 1)).astype(np.float32))


class LogisticReg(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.lin = nn.Linear(input_dim, output_dim)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return torch.sigmoid(self.lin(x))


n_data, input_dim = x_np.shape
_, n_classes = y_np.shape

m = LogisticReg(input_dim, n_classes)
opt = torch.optim.SGD(m.parameters(), lr=lr)
l = nn.BCELoss()

for ep in range(epochs):
    y_pred = m(x_train)

    loss_i = l(y_pred, y_train)

    if loss_i < train_stop_loss_th:
        print(f"training stopped at epoch #{ep}")
        break

    

    opt.zero_grad()
    loss_i.backward()
    # if ep % 5 == 0:
    #     w, b = m.parameters()

    #     print(f"w: {w[0][0]} | b: {b[0]}")
    #     print(f"w_grad: {w.grad} | b: {b.grad}")

    opt.step()

    if ep % 5 == 0:
        acc_i = y_pred.round().eq(y_train).sum().item() / len(y_train)
        
        print(f"epoch: {ep} | loss: {loss_i} | acc: {acc_i}")

        # print()


preds = m(x_test)
accuracy = preds.round().eq(y_test).sum() / len(preds)
print(f"accruacy: {accuracy}")
