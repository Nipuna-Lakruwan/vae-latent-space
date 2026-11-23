"""Figures and GIFs for the README.

    uv run python src/figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.stats import norm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from common import CKPT, CLASSES, RESULTS, device, fashion
from vae import VAE

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def load(z, dev):
    s = torch.load(CKPT / f"vae_z{z}.pt", map_location=dev)
    m = VAE(s["z_dim"]).to(dev)
    m.load_state_dict(s["model"])
    return m.eval()


def grid(x, nrow, pad=1):
    return make_grid(x, nrow=nrow, padding=pad, pad_value=0.3)[0].cpu().numpy()


@torch.no_grad()
def latent_map(dev):
    m = load(2, dev)
    n = 20
    qs = norm.ppf(np.linspace(0.03, 0.97, n))  # evenly spaced in probability under the prior
    zz = torch.tensor([[a, b] for b in qs[::-1] for a in qs], dtype=torch.float32, device=dev)
    imgs = torch.sigmoid(m.decode(zz))
    te = DataLoader(fashion(False), batch_size=1000)
    mus, ys = [], []
    for x, y in te:
        mus.append(m.encode(x.to(dev))[0].cpu()); ys.append(y)
    mus, ys = torch.cat(mus).numpy(), torch.cat(ys).numpy()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.3))
    cmap = plt.get_cmap("tab10")
    for c in range(10):
        s = ys == c
        a1.scatter(mus[s, 0], mus[s, 1], s=2, alpha=0.5, color=cmap(c), label=CLASSES[c])
    a1.legend(markerscale=6, fontsize=7, loc="upper right", ncol=2)
    a1.set_title("Where 10,000 test images land in the 2-D latent space")
    a1.set_xlabel("$z_1$"); a1.set_ylabel("$z_2$"); a1.set_xlim(-3.5, 3.5); a1.set_ylim(-3.5, 3.5)
    a2.imshow(grid(imgs, n, 0), cmap="gray")
    a2.set_title("Decoder output for a 20x20 grid of latent points"); a2.axis("off")
    fig.tight_layout(); fig.savefig(RESULTS / "latent_map_z2.png"); plt.close(fig)


@torch.no_grad()
def interpolation(dev, z=16):
    m = load(z, dev)
    ds = fashion(False)
    # pick the first test image of each class in a loop: sneaker -> boot -> sandal -> bag -> trouser -> dress -> T-shirt
    order = [7, 9, 5, 8, 1, 3, 0, 7]
    first = {}
    for i in range(len(ds)):
        y = ds[i][1]
        first.setdefault(y, ds[i][0])
        if len(first) == 10:
            break
    xs = torch.stack([first[c] for c in order]).to(dev)
    mu = m.encode(xs)[0]
    frames, steps = [], 16
    for a, b, xa, xb, ca, cb in zip(mu[:-1], mu[1:], xs[:-1], xs[1:], order[:-1], order[1:]):
        for t in np.linspace(0, 1, steps, endpoint=False):
            zt = (1 - t) * a + t * b
            lat = torch.sigmoid(m.decode(zt[None]))[0, 0].cpu().numpy()
            pix = ((1 - t) * xa + t * xb)[0].cpu().numpy()
            frames.append((lat, pix, f"{CLASSES[ca]} -> {CLASSES[cb]}"))
    imgs = []
    for lat, pix, label in frames:
        tile = lambda a: Image.fromarray((a * 255).astype(np.uint8)).resize((168, 168), Image.NEAREST)
        canvas = Image.new("RGB", (2 * 168 + 30, 168 + 50), (14, 14, 18))
        canvas.paste(tile(lat), (10, 40)); canvas.paste(tile(pix), (168 + 20, 40))
        d = ImageDraw.Draw(canvas)
        d.text((10, 6), label, fill=(235, 235, 235))
        d.text((10, 22), "latent space", fill=(140, 200, 255)); d.text((188, 22), "pixel space (fade)", fill=(255, 170, 120))
        imgs.append(canvas)
    imgs[0].save(RESULTS / "interpolation.gif", save_all=True, append_images=imgs[1:], duration=70, loop=0)
    # static strip: one transition, latent vs pixel
    sel = [k for k in range(steps)][::2] + [steps]
    a, b = mu[0], mu[1]
    lat = torch.sigmoid(m.decode(torch.stack([(1 - t) * a + t * b for t in np.linspace(0, 1, 9)])))
    pix = torch.stack([(1 - t) * xs[0] + t * xs[1] for t in np.linspace(0, 1, 9)])
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.imshow(grid(torch.cat([lat, pix]), 9, 2), cmap="gray")
    ax.set_yticks([15, 45], ["latent", "pixel fade"]); ax.set_xticks([])
    ax.set_title("Sneaker -> ankle boot: walking through latent space vs. blending pixels")
    fig.tight_layout(); fig.savefig(RESULTS / "interpolation_strip.png"); plt.close(fig)


@torch.no_grad()
def recon_and_samples(dev, zs=(2, 16, 32)):
    ds = fashion(False)
    x = torch.stack([ds[i][0] for i in range(12)]).to(dev)
    rows = [x]
    for z in zs:
        m = load(z, dev)
        rows.append(torch.sigmoid(m.decode(m.encode(x)[0])))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.imshow(grid(torch.cat(rows), 12, 2), cmap="gray")
    ax.set_yticks([15 + 30 * i for i in range(len(rows))], ["input"] + [f"z = {z}" for z in zs]); ax.set_xticks([])
    ax.set_title("Reconstructions: a bigger latent keeps more detail")
    fig.tight_layout(); fig.savefig(RESULTS / "reconstructions.png"); plt.close(fig)

    g = torch.Generator(device=dev).manual_seed(0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.0))
    for ax, z in zip(axes, (16, 32)):
        m = load(z, dev)
        s = torch.sigmoid(m.decode(torch.randn(64, z, device=dev, generator=g)))
        ax.imshow(grid(s, 8), cmap="gray"); ax.axis("off"); ax.set_title(f"New items: z ~ N(0, I), z = {z}")
    fig.tight_layout(); fig.savefig(RESULTS / "samples.png"); plt.close(fig)


def curves():
    d = json.loads((RESULTS / "metrics.json").read_text())
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    for k, v in d.items():
        h = v["history"]
        ax.plot([r["epoch"] for r in h], [r["test_neg_elbo"] for r in h], label=f"{k.replace('z', 'z = ')}")
    ax.set_xlabel("epoch"); ax.set_ylabel("test -ELBO (nats / image)"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Lower is better")
    fig.tight_layout(); fig.savefig(RESULTS / "elbo_curves.png"); plt.close(fig)


@torch.no_grad()
def active_dims(dev, zs=(2, 16, 32), thresh=0.05):
    """Average KL per latent dimension on the test set; dimensions near 0 are unused (collapsed to the prior)."""
    x = torch.cat([a for a, _ in DataLoader(fashion(False), batch_size=2000)]).to(dev)
    out = {}
    for z in zs:
        mu, lv = load(z, dev).encode(x)
        kl = (0.5 * (mu ** 2 + lv.exp() - lv - 1)).mean(0).cpu()
        out[f"z{z}"] = {"active": int((kl > thresh).sum()), "kl_per_dim": sorted(kl.tolist(), reverse=True)}
    (RESULTS / "active_dims.json").write_text(json.dumps(out, indent=2))


def main():
    dev = device()
    curves()
    latent_map(dev)
    interpolation(dev)
    recon_and_samples(dev)
    active_dims(dev)


if __name__ == "__main__":
    main()
