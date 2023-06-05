import torch
import torch.nn as nn
import numpy as np

l = nn.CrossEntropyLoss()
Y = torch.tensor([0])

y_pred_good = torch.tensor([[2.0, 0.1, 0.1]])
y_pred_bad = torch.tensor([[0.1, 2.0, 0.1]])

print(f'good loss: {l(y_pred_good, Y)}')
print(f'bad loss: {l(y_pred_bad, Y)}')