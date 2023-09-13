import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, n_channels, h_dim, decoder_h_dim, z_dim):
        super(AutoEncoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(n_channels, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim)
        )

        self.decode = nn.Sequential(
            nn.Linear(z_dim, decoder_h_dim),
            nn.ReLU(),
            nn.Linear(decoder_h_dim, decoder_h_dim),
            nn.ReLU(),
            nn.Linear(decoder_h_dim, decoder_h_dim),
            nn.ReLU(),
            nn.Linear(decoder_h_dim, n_channels),
            nn.Sigmoid()
        )

    def latent(self, x):
        return self.encode(x)

    def penalty(self):
        return None

    def forward(self, x):
        z = self.latent(x)
        x = self.decode(z)
        return z, x
