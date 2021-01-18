from torch import nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.latent_dim=latent_dim
        self.img_shape=img_shape

    def forward(self, z):
        return self.model(z)
