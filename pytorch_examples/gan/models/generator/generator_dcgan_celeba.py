import torch
from torch import nn

from pytorch_examples.gan.models import DiscriminatorDCGAN_CELEBA
from pytorch_examples.gan.models import Generator
from pytorch_examples.gan.models import LambdaModule


class GeneratorDCGAN_CELEBA(Generator):
    def __init__(self, latent_dim, img_shape,ngf=64):
        super().__init__(latent_dim, img_shape)
        nc=img_shape[0]
        self.model = nn.Sequential(
            LambdaModule(lambda x: x),
            # input is Z, going into a convolution
            LambdaModule(lambda x: x.unsqueeze(-1).unsqueeze(-1)),
            nn.ConvTranspose2d( latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.apply(self.weights_init)

    # custom weights initialization called on netG and netD
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

if __name__== '__main__':
    from torchsummary import summary
    latent_dim=100
    img_shape=(3,64,64)
    input = torch.randn(64, latent_dim)
    netG = GeneratorDCGAN_CELEBA(latent_dim, img_shape)
    out_g = netG(input)
    assert out_g.shape[1:]==img_shape
    netD = DiscriminatorDCGAN_CELEBA(img_shape)
    out_d = netD(out_g)
    assert out_d.shape[1:]==(1,)

    summary(netG, (latent_dim,), batch_size=1)
    summary(netD, img_shape, batch_size=1)

