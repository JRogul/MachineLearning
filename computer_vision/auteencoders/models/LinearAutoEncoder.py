import torch.nn as nn
import torch

class LinearAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 784),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        x = self.decoder(x)

        return torch.sigmoid(x)

