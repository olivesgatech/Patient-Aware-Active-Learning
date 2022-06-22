import torch.nn as nn


class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.layer = nn.Linear(n_inputs, 3)

    # forward propagate input
    def forward(self, X):
        X = self.layer(X)
        return X