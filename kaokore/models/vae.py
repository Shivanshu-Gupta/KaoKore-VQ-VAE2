import abc
import numpy as np
import logging
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchinfo import summary

from models.utils import ResBlock, add_res_block, add_conv2d_block, add_convtranspose2d_block
from models.nearest import NearestEmbed, Quantize

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

class CVAE(AbstractAutoEncoder):
    """
    Convolutional VAE
    - Gaussian Latent
    - loss = mse reconstruction error + KL Divergence
    """
    def __init__(self, d, kl_coef=0.1, num_channels=3, bn=True, input_size=256, **kwargs):
        super(CVAE, self).__init__()
        self.f = 64 if input_size == 256 else 8
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
        breakpoint()
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
        breakpoint()
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
    def __init__(self, d, k=10, bn=True, vq_coef=1, commit_coef=0.5, num_channels=3, **kwargs):
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

    def loss(self, x, recon_x, z_e, emb, argmin):
        self.mse = F.mse_loss(recon_x, x)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1))
        self.commit_loss = torch.mean(torch.norm((emb.detach() - z_e)**2, 2, 1))

        return self.mse + self.vq_coef * self.vq_loss + self.commit_coef * self.commit_loss

    def latest_losses(self):
        return {'mse': self.mse, 'vq': self.vq_loss, 'commitment': self.commit_loss}

    @classmethod
    def test(_):
        vqcvae = VQ_CVAE(d=256)
        summary(vqcvae, input_size=(64, 3, 32, 32), depth=4)
        x = torch.rand(64, 3, 32, 32)
        _x, mu, logvar, argmin = vqcvae(x)
        loss = vqcvae.loss(x, _x, mu, logvar)

def get_encoder(in_channel, channel, n_res_block, res_channel, stride, bn=False):
    encoder_layers = []
    add_conv2d_block(encoder_layers, in_channel, channel // 2, 4, 2, 1, bn=bn)
    if stride == 4:
        add_conv2d_block(encoder_layers, channel // 2, channel, 4, 2, 1, bn=bn)
        encoder_layers.append(nn.Conv2d(channel, channel, 3, padding=1))
    else:
        encoder_layers.append(nn.Conv2d(channel // 2, channel, 3, padding=1))
    for _ in range(n_res_block):
        add_res_block(encoder_layers, channel, channel, res_channel, bn=bn, final_bn=bn)
    encoder_layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*encoder_layers)

def get_decoder(in_channel, out_channel, channel, n_res_block, res_channel, stride, bn=False):
    decoder_layers = [nn.Conv2d(in_channel, channel, 3, padding=1)]
    for _ in range(n_res_block):
        add_res_block(decoder_layers, channel, channel, res_channel, bn=bn, final_bn=bn)
    if stride == 4:
        add_convtranspose2d_block(decoder_layers, channel, channel // 2, 4, 2, 1, bn=bn)
        decoder_layers.append(nn.ConvTranspose2d(channel // 2, out_channel, kernel_size=4, stride=2, padding=1))
    else:
        decoder_layers.append(nn.ConvTranspose2d(channel, out_channel, kernel_size=4, stride=2, padding=1))
    return nn.Sequential(*decoder_layers)

class VQ_VAE2(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        res_channel=32,
        embed_dim=64,
        n_embed=512,
        bn=False,
        decay=0.99,
        commit_coef=0.25,
    ):
        super().__init__()

        # ENCODER: input -> (quant_b, quant_t)
        # input -> enc_b -> enc_t -> conv_t -> quant_t -> dec_t
        self.enc_b = get_encoder(in_channel, channel, n_res_block, res_channel, stride=4, bn=bn)
        self.enc_t = get_encoder(channel, channel, n_res_block, res_channel, stride=2, bn=bn)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = get_decoder(embed_dim, embed_dim, channel,
                                 n_res_block, res_channel, stride=2, bn=bn)

        # (enc_b, dec_t) -> conv_b -> quant_b
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)

        # DECODER: (quant_b, quant_t) -> output
        # quant_t -> unsample_t
        self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)

        # (upsample_t, quant_b) -> dec_b (output)
        self.dec_b = get_decoder(embed_dim + embed_dim, in_channel, channel,
                                 n_res_block, res_channel, stride=4, bn=bn)

        self.commit_coef = commit_coef

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec_b = self.dec_b(quant)

        return dec_b

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec

    def loss(self, x, x_rec, diff, return_parts=False):
        recon_loss = F.mse_loss(x_rec, x, reduction='mean')
        commit_loss = diff.mean()
        loss = recon_loss + self.commit_coef * commit_loss
        if not return_parts: return loss
        else: return loss, recon_loss, commit_loss

    @classmethod
    def test(_):
        vqvae2 = VQ_VAE2()
        input_size=(64, 3, 256, 256)
        summary(vqvae2, input_size=input_size, depth=4)
        x = torch.rand(*input_size)
        _x, diff = vqvae2(x)
        loss = vqvae2.loss(x, _x, diff)
