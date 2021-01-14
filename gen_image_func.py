import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import random
import torchvision.utils as vutils

class Generator(nn.Module):
    def __init__(self, ngpu):
        ngf = 64
        nz = 100
        nc = 3
        ngpu = 0
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
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

    def forward(self, input):
        return self.main(input)


def generate():
    filename = 'generator_model_50.h5'
    net = torch.load(filename, map_location=torch.device('cpu'))
    nz = 100
    ngpu = 0
    device = torch.device("cpu")
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    imgs = net(fixed_noise).detach().cpu()

    plt.axis("off")
    fixed_noise = torch.randn(1, nz, 1, 1, device=device)
    r = random.randrange(64)
    img_random = np.transpose(vutils.make_grid(imgs[1].to(device), normalize=True).cpu(), (1, 2, 0))
    plt.imshow(img_random)
    plt.savefig('static/images/img.jpg')
