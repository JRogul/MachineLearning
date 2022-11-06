import torch.nn as nn
import torch

class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU())

        self.mu = nn.Linear(128, 64)
        self.sigma = nn.Linear(128, 64)

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        mu, sigma = self.mu(x), self.sigma(x)
        eps = torch.randn_like(sigma)
        reparametrized = mu + sigma * eps

        x = self.decoder(reparametrized)
        return x, mu, sigma

