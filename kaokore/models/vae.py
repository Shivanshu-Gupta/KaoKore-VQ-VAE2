import abc
import numpy as np
import logging
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchinfo import summary

from nearest import NearestEmbed

class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return

class VAE(AbstractAutoEncoder):
    """
    Variational AutoEncoder for MNIST
    - Gaussian Latent
    - loss = bce reconstruction error + KL divergence
    Taken from pytorch/examples: https://github.com/pytorch/examples/tree/master/vae
    """

    def __init__(self, kl_coef=1, in_dim=784, z_dim=20, **kwargs):
        super(VAE, self).__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.fc1 = nn.Linear(in_dim, 400)
        self.fc21 = nn.Linear(400, z_dim)
        self.fc22 = nn.Linear(400, z_dim)
        self.fc3 = nn.Linear(z_dim, 400)
        self.fc4 = nn.Linear(400, in_dim)

        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.kl_coef = kl_coef
        self.bce = 0
        self.kl = 0

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            # TODO: rewrite this
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.fc3(z).relu()
        return self.fc4(h3).sigmoid()

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.in_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, size):
        z = torch.randn(size, self.z_dim)
        if self.cuda(): z = z.cuda()
        x = self.decode(z).cpu()
        return x

    def loss(self, x, recon_x, mu, logvar):
        self.bce = F.binary_cross_entropy(
            recon_x, x.view(-1, self.in_dim), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return self.bce + self.kl_coef * self.kl

    def latest_losses(self):
        return {'bce': self.bce, 'kl': self.kl}

    @classmethod
    def test(_):
        cvae = VAE()
        x = torch.rand(64, 1, 28, 28)
        _x, mu, logvar = cvae(x)
        loss = cvae.loss(x, _x, mu, logvar)

class VQ_VAE(AbstractAutoEncoder):
    """
        Vector Quantized AutoEncoder for mnist
        - Categorical Latent
        - loss = bce reconstruction error + codebook loss + commitment loss
    """

    def __init__(self, in_dim=784, z_dim=200, k=10, vq_coef=0.2, comit_coef=0.4, **kwargs):
        super(VQ_VAE, self).__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.emb_dim = k
        assert z_dim % k == 0

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 400),
            nn.ReLU(),
            nn.Linear(400, z_dim))
        self.emb = NearestEmbed(k, self.emb_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 400),
            nn.ReLU(),
            nn.Linear(400, in_dim),
            nn.Sigmoid())
        # self.fc1 = nn.Linear(in_dim, 400)
        # self.fc2 = nn.Linear(400, z_dim)
        # self.fc3 = nn.Linear(z_dim, 400)
        # self.fc4 = nn.Linear(400, in_dim)

        self.vq_coef = vq_coef
        self.comit_coef = comit_coef
        self.ce_loss = self.vq_loss = self.commit_loss = 0

    def encode(self, x):
        # h1 = self.relu(self.fc1(x))
        # z = self.fc2(h1)
        z = self.encoder(x)
        return z.view(-1, self.emb_dim, self.z_dim // self.emb_dim)

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        # _x = F.tanh(self.fc4(h3))
        _x = self.decoder(z)
        return _x

    def forward(self, x):
        """
            Args:
                x: (batch_size, *, in_dim)
                    Input Image

            Returns:
                recon_x: (batch_size, *, in_dim)
                    Reconstructed Image
                z_e: (batch_size, *, emb_dim, z_dim // emb_dim)
                    Encoder Output
                emb: (batch_size, *, emb_dim, z_dim // emb_dim)
                    Quantized Encoder Output
        """
        z_e = self.encode(x.view(-1, self.in_dim))  # (batch_size, emb_dim, z_dim/emb_dim) - latent encoder output
        z_q = self.emb(z_e, emb_sg=True).view(-1, self.z_dim) # (batch_size, z_dim) - quantized latent (decoder input)
        emb = self.emb(z_e.detach()) # (batch_size, z_dim) - quantized latent
        recon_x = self.decode(z_q)  # (batch_size, in_dim) - reconstructed image
        return recon_x, z_e, emb

    def sample(self, size):
        """
            Sample from the VQ-VAE

            Args:
                size (int): number of sample images

            Returns:
                x: (size, in_dim)
        """

        z = torch.randn(size, self.emb_dim, self.z_dim // self.emb_dim)
        if self.cuda(): z = z.cuda()
        emb = self.emb(z).view(-1, self.z_dim)
        x = self.decode(emb).cpu()
        return x

    def loss(self, x, recon_x, z_e, emb):
        self.ce_loss = F.binary_cross_entropy(recon_x, x.view(-1, self.in_dim)) # Reconstruction Loss
        self.vq_loss = F.mse_loss(emb, z_e.detach())                            # VQ Loss - update the embedding table
        self.commit_loss = F.mse_loss(emb.detach(), z_e)                        # Commitment Loss - update the encoder

        return self.ce_loss + self.vq_coef * self.vq_loss + self.comit_coef * self.commit_loss

    def latest_losses(self):
        return {'cross_entropy': self.ce_loss, 'vq': self.vq_loss, 'commitment': self.commit_loss}

    @classmethod
    def test(_):
        vqvae = VQ_VAE()
        x = torch.randint(0, 2, (64, 784), dtype=torch.float32)
        _x, z_e, emb = vqvae(x)
        loss = vqvae.loss(x, _x, z_e, emb)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()
        mid_channels = mid_channels or out_channels
        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        ]
        if bn: layers.insert(2, nn.BatchNorm2d(mid_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)

def add_res_block(layers, in_channels, out_channels, mid_channels=None, bn=False, final_bn=False):
    layers.append(ResBlock(in_channels, out_channels, mid_channels, bn))
    if final_bn: layers.append(nn.BatchNorm2d(out_channels))

def add_conv2d_block(layers, in_channels, out_channels, kernel_size, stride=1, padding=0,
                   activation=nn.ReLU(inplace=True), bn=False):
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    if bn: layers.append(nn.BatchNorm2d(out_channels))
    layers.append(activation)

def add_convtranspose2d_block(layers, in_channels, out_channels, kernel_size, stride=1, padding=0,
                   activation=nn.ReLU(inplace=True), bn=False):
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
    if bn: layers.append(nn.BatchNorm2d(out_channels))
    layers.append(activation)

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 activation=nn.ReLU(inplace=True), bn=False):
        super(Conv2dBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            activation
        ]
        if bn: layers.insert(1, nn.BatchNorm2d(out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 activation=nn.ReLU(inplace=True), bn=False):
        super(ConvTranspose2dBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            activation
        ]
        if bn: layers.insert(1, nn.BatchNorm2d(out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CVAE(AbstractAutoEncoder):
    """
    Convolutional VAE
    - Gaussian Latent
    - loss = mse reconstruction error + KL Divergence
    """
    def __init__(self, d, kl_coef=0.1, num_channels=3, bn=True, **kwargs):
        super(CVAE, self).__init__()
        self.f = 8  # TODO: make this a parameter if possible
        self.d = d

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),

            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            ResBlock(d, d, bn=True)
        )
        self.fc11 = nn.Linear(d * self.f ** 2, d * self.f ** 2)
        self.fc12 = nn.Linear(d * self.f ** 2, d * self.f ** 2)
        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d//2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(d // 2, num_channels, kernel_size=4, stride=2, padding=1, bias=False)
        )
        self.kl_coef = kl_coef
        self.kl_loss = self.mse = 0

    def encode(self, x):
        h1 = self.encoder(x)
        h1 = h1.view(-1, self.d * self.f ** 2)
        return self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return h3.tanh()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, size):
        sample = torch.randn(size, self.d * self.f ** 2, requires_grad=False)
        if self.cuda(): sample = sample.cuda()
        return self.decode(sample).cpu()

    def loss(self, x, recon_x, mu, logvar):
        self.mse = F.mse_loss(recon_x, x)
        batch_size = x.size(0)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        self.kl_loss /= batch_size * 3 * 1024

        # return mse
        return self.mse + self.kl_coef * self.kl_loss

    def latest_losses(self):
        return {'mse': self.mse, 'kl': self.kl_loss}

    @classmethod
    def test(_):
        cvae = CVAE(d=256)
        x = torch.rand(64, 3, 32, 32)
        _x, mu, logvar = cvae(x)
        loss = cvae.loss(x, _x, mu, logvar)

class VQ_CVAE(AbstractAutoEncoder):
    """
    Vector Quantized AutoEncoder
    - Catetorical Latent
    - loss = mse reconstruction error + codebook loss + commitment loss
    """
    def __init__(self, d, k=10, bn=True, vq_coef=1, commit_coef=0.5, num_channels=3, old=True, **kwargs):
        super(VQ_CVAE, self).__init__()
        self.d = d

        # TODO: check whether to use d // 2 instead of d
        encoder_layers = []
        add_conv2d_block(encoder_layers, num_channels, d, 4, 2, 1, bn=bn)
        add_conv2d_block(encoder_layers, d, d, 4, 2, 1, bn=bn)
        add_res_block(encoder_layers, d, d, bn=bn, final_bn=bn)
        add_res_block(encoder_layers, d, d, bn=bn, final_bn=bn)
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        add_res_block(decoder_layers, d, d, bn=False, final_bn=bn)
        add_res_block(decoder_layers, d, d, bn=False, final_bn=bn)
        add_convtranspose2d_block(decoder_layers, d, d, 4, 2, 1, bn=bn)
        decoder_layers.append(nn.ConvTranspose2d(d, num_channels, kernel_size=4, stride=2, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)

        self.emb = NearestEmbed(k, d)

        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = self.vq_loss = self.commit_loss = 0

        for l in self.modules():
            # TODO: Figure out what's going on here
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        if bn: self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.embs.detach().normal_(0, 0.02)
        torch.fmod(self.emb.embs, 0.04)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder(x))

    def forward(self, x):
        """
            Args:
                x: (batch_size, num_channels, height, width)
                    Input Image

            Returns:
                _x: (batch_size, num_channels, height, width)
                    Reconstructed Image
                z_e: (batch_size, d, f, f)
                    Encoder Output
                emb: (batch_size, d, f, f)
                    Quantized Encoder Output
        """
        # breakpoint()
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, emb_sg=True, return_argmin=True)
        emb = self.emb(z_e.detach())
        _x = self.decode(z_q)
        return _x, z_e, emb, argmin

    def sample(self, size):
        """
            Sample from the VQ-VAE

            Args:
                size (int): number of sample images

            Returns:
                x: (size, num_channels, height, width)
        """
        sample = torch.randn(size, self.d, self.f, self.f,
                             requires_grad=False),
        if self.cuda(): sample = sample.cuda()
        emb, _ = self.emb(sample)
        x =  self.decode(emb.view(size, self.d, self.f, self.f)).cpu()
        return x

    def loss(self, x, recon_x, z_e, emb):
        self.mse = F.mse_loss(recon_x, x)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1))
        self.commit_loss = torch.mean(torch.norm((emb.detach() - z_e)**2, 2, 1))

        return self.mse + self.vq_coef * self.vq_loss + self.commit_coef * self.commit_loss

    def latest_losses(self):
        return {'mse': self.mse, 'vq': self.vq_loss, 'commitment': self.commit_loss}

    def print_atom_hist(self, argmin):
        argmin = argmin.detach().cpu().numpy()
        unique, counts = np.unique(argmin, return_counts=True)
        logging.info(counts)
        logging.info(unique)

    @classmethod
    def test(_):
        vqcvae = VQ_CVAE(d=256)
        summary(vqcvae, input_size=(64, 3, 32, 32), depth=4)
        x = torch.rand(64, 3, 32, 32)
        _x, mu, logvar, argmin = vqcvae(x)
        loss = vqcvae.loss(x, _x, mu, logvar)
