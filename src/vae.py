"""Convolutional VAE (Kingma & Welling 2014).

Encoder q_phi(z|x) = N(mu(x), diag sigma^2(x));  prior p(z) = N(0, I);  decoder p_theta(x|z) = Bernoulli(logits(z))

ELBO(x) = E_q[ log p(x|z) ] - KL( q(z|x) || p(z) )
        = -BCE(x, decoder(z))  - 1/2 * sum( mu^2 + sigma^2 - log sigma^2 - 1 )

Reparameterisation trick: z = mu + sigma * eps, eps ~ N(0, I), so gradients flow through mu and sigma.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, z_dim=16):
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.SiLU(),        # 14x14
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),       # 7x7
            nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),      # 4x4
            nn.Flatten(), nn.Linear(128 * 16, 512), nn.SiLU())
        self.mu = nn.Linear(512, z_dim)
        self.logvar = nn.Linear(512, z_dim)
        self.dec_in = nn.Sequential(nn.Linear(z_dim, 512), nn.SiLU(), nn.Linear(512, 128 * 16), nn.SiLU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1), nn.SiLU(),        # 7x7
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),         # 14x14
            nn.ConvTranspose2d(32, 1, 4, 2, 1))                     # 28x28 logits

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h).clamp(-12, 8)

    def decode(self, z):
        return self.dec(self.dec_in(z).view(-1, 128, 4, 4))  # logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)  # reparameterisation trick
        return self.decode(z), mu, logvar


def elbo_terms(model, x):
    logits, mu, logvar = model(x)
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1)   # -log p(x|z)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1)
    return rec, kl


@torch.no_grad()
def iwae_bound(model, x, k=100):
    """Importance-weighted estimate of log p(x) (Burda et al. 2016), tighter than the ELBO:
        log p(x) >= log 1/k sum_i p(x|z_i) p(z_i) / q(z_i|x),  z_i ~ q(z|x)"""
    mu, logvar = model.encode(x)
    std = (0.5 * logvar).exp()
    ws = []
    for _ in range(k):
        eps = torch.randn_like(mu)
        z = mu + std * eps
        log_px_z = -F.binary_cross_entropy_with_logits(model.decode(z), x, reduction="none").flatten(1).sum(1)
        log_pz = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(1)
        log_qz = -0.5 * (eps.pow(2) + logvar + math.log(2 * math.pi)).sum(1)
        ws.append(log_px_z + log_pz - log_qz)
    ws = torch.stack(ws, 0)
    return torch.logsumexp(ws, 0) - math.log(k)
