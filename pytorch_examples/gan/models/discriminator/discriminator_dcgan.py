from torch import nn
import torch
from pytorch_examples.gan.models.discriminator.discriminator_abstract import Discriminator
from pytorch_examples.gan.models.utils import LambdaModule


class DiscriminatorDCGAN(Discriminator):
    def __init__(self, img_shape ,ndf = 64,):
        super().__init__(img_shape)
        nc=img_shape[0]
        self.model = nn.Sequential(
            # input is (nc) x 32 x 32
            LambdaModule(lambda x: x.view(x.size(0),nc,32,32)),
            nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            nn.Flatten(1)
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
    img_shape = (3,32,32)
    input = torch.randn(64,*img_shape)
    netD = DiscriminatorDCGAN(img_shape)
    netD.apply(DiscriminatorDCGAN.weights_init)
    out = netD(input)
    assert out.shape[1:] == (1,)
    summary(netD, img_shape, batch_size=32)
