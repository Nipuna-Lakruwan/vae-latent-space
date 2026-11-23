"""Train VAEs with several latent sizes on Fashion-MNIST and evaluate them on the test set.

    uv run python src/train.py --z 2 16 32 --epochs 30
"""
import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from common import CKPT, RESULTS, device, fashion
from vae import VAE, elbo_terms, iwae_bound


def evaluate(model, loader, dev, iwae_k=0):
    model.eval()
    tot = {"rec": 0.0, "kl": 0.0, "iwae": 0.0, "n": 0}
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(dev)
            rec, kl = elbo_terms(model, x)
            tot["rec"] += rec.sum().item(); tot["kl"] += kl.sum().item(); tot["n"] += len(x)
            if iwae_k:
                tot["iwae"] += iwae_bound(model, x, iwae_k).sum().item()
    n = tot.pop("n")
    out = {k: v / n for k, v in tot.items()}
    out["neg_elbo"] = out["rec"] + out["kl"]
    if iwae_k:
        out["neg_iwae"] = -out.pop("iwae")
    else:
        out.pop("iwae")
    model.train()
    return out


def train_one(z_dim, epochs, dev, seed=0):
    torch.manual_seed(seed)
    tr = DataLoader(fashion(True), batch_size=128, shuffle=True, drop_last=True)
    te = DataLoader(fashion(False), batch_size=500)
    model = VAE(z_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = []
    for ep in range(epochs):
        t0, run = time.time(), 0.0
        for x, _ in tr:
            x = x.to(dev)
            rec, kl = elbo_terms(model, x)
            loss = (rec + kl).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item()
        sched.step()
        ev = evaluate(model, te, dev)
        hist.append({"epoch": ep + 1, "train_neg_elbo": run / len(tr), **{f"test_{k}": v for k, v in ev.items()}})
        print(f"z={z_dim} epoch {ep + 1}: train -ELBO {run / len(tr):.1f}  test -ELBO {ev['neg_elbo']:.1f} "
              f"(rec {ev['rec']:.1f} + KL {ev['kl']:.1f})  {time.time() - t0:.0f}s", flush=True)
    final = evaluate(model, te, dev, iwae_k=100)
    torch.save({"model": model.state_dict(), "z_dim": z_dim}, CKPT / f"vae_z{z_dim}.pt")
    return hist, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=int, nargs="+", default=[2, 16, 32])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    dev = device()
    CKPT.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    out_p = RESULTS / "metrics.json"
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    for z in args.z:
        hist, final = train_one(z, args.epochs, dev)
        final["bits_per_dim_iwae"] = final["neg_iwae"] / (784 * 0.6931471805599453)
        out[f"z{z}"] = {"final": final, "history": hist}
        print(f"z={z} final test: {final}", flush=True)
        out_p.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
