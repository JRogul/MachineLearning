import torch
import torch.nn as nn

class VariationalCNNAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mu = nn.Linear(3136, 2)
        self.sigma = nn.Linear(3136, 2)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten()

        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 3136),
            nn.Unflatten(dim=1, unflattened_size=(64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2,padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2,padding=1),
            nn.Sigmoid()            
            
        )


    def forward(self, x):
        x = self.encoder(x)
        mu, sigma = self.mu(x), self.sigma(x)
        eps = torch.rand_like(sigma)
        
        reparametrized = mu + sigma * eps
        x = self.decoder(reparametrized)
        x = x[:, :, :28, :28]
        
        return x, mu, sigma



if '__init__' == '__main__':
    model = VariationalCNNAutoEncoder()
    print(model(torch.randn(128,1,28,28))[0].shape)

