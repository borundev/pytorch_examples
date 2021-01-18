import numpy as np
import torch.nn as nn
import torch
from pytorch_examples.gan.models import Discriminator
from pytorch_examples.gan.models import LambdaModule


class DiscriminatorFF(Discriminator):
    def __init__(self, img_shape):
        super().__init__(img_shape)
        self.model = nn.Sequential(
            LambdaModule(lambda img:img.view(img.size(0), -1)),
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )


if __name__== '__main__':
    from torchsummary import summary
    img_shape = (3,64,64)
    input = torch.randn(64,*img_shape)
    netD = DiscriminatorFF(img_shape)
    out = netD(input)
    assert out.shape[1:] == (1,)
    summary(netD, img_shape, batch_size=32)
