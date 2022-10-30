import torch.nn as nn


class AutoEncoder_mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(True),
            nn.Flatten(),
            nn.Linear(192, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 192),
            nn.BatchNorm1d(192),
            nn.Unflatten(dim=1, unflattened_size=(12, 4, 4)),
            nn.ConvTranspose2d(in_channels=12, out_channels=6, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

