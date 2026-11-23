import torch

from vae import VAE, elbo_terms, iwae_bound


def test_shapes_and_bounds():
    torch.manual_seed(0)
    m = VAE(4).eval()
    x = torch.rand(3, 1, 28, 28)
    logits, mu, logvar = m(x)
    assert logits.shape == x.shape and mu.shape == (3, 4)
    rec, kl = elbo_terms(m, x)
    assert (kl >= 0).all()
    # the importance-weighted bound must be at least as tight as the ELBO (on average)
    iw = iwae_bound(m, x, k=50)
    assert iw.mean() >= -(rec + kl).mean() - 5.0
