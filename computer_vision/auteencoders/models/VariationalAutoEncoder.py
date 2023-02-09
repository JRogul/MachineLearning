import torch.nn as nn
import torch

class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU()),

        self.mu = nn.Linear(128, 2)
        self.sigma = nn.Linear(128, 2)

        self.decoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.Sigmoid()
        )
    def encode(self, x):
        x = self.encoder(x)
        mu, sigma = self.mu(x), self.sigma(x)
        eps = torch.randn_like(sigma)
        reparametrized = mu + sigma * eps
        return reparametrized, mu, sigma
    def forward(self, x):
        x, mu, sigma = self.encode(x)
        x = self.decoder(x)
        return x, mu, sigma

