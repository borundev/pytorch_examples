import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.img_shape=img_shape

    def forward(self, img):
        raise NotImplementedError()