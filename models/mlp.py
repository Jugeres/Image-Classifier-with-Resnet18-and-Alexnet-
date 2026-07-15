import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Forward propagation
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return out
