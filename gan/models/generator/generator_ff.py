import numpy as np
import torch.nn as nn
import torch

from gan.models.generator.generator_abstract import Generator
from gan.models.discriminator import DiscriminatorFF
from gan.models.utils import LambdaModule


class GeneratorFF(Generator):
    def __init__(self, latent_dim, img_shape):
        super().__init__(latent_dim, img_shape)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
            LambdaModule(lambda img:img.view(img.size(0), *self.img_shape))
        )



if __name__== '__main__':
    from torchsummary import summary
    latent_dim=128
    img_shape=(3,32,32)
    input = torch.randn(64, latent_dim)
    netG = GeneratorFF(latent_dim, img_shape)
    out_g = netG(input)
    assert out_g.shape[1:]==img_shape
    netD = DiscriminatorFF(img_shape)
    out_d = netD(out_g)
    assert out_d.shape[1:]==(1,)

    summary(netG, (latent_dim,), batch_size=1)
    summary(netD, img_shape, batch_size=1)