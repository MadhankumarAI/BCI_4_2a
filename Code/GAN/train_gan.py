"""
Train the per-subject EEG-GANs and cache their synthetic trials.
================================================================

Run this before Code/GAN/main.py. For each subject it trains

    1 + n_folds  generators

    * the "full" generator, fitted on all of that subject's T-session
      Left/Right trials. This is the one whose samples go into the final
      model.
    * one generator per gate fold, fitted on that fold's TRAINING portion
      only. These exist purely so the augmentation-selection gate in main.py
      is leak-free.

Why the per-fold generators are not optional
--------------------------------------------
The obvious shortcut is to train one GAN on the whole training set and then
cross-validate the augmentation ratio on that same training set. That leaks:
the synthetic trials added to fold k's training portion were produced by a
generator that had already seen fold k's validation trials, so the gate is
scoring a model that has indirect knowledge of its own validation data. The
gate would then reliably choose "more synthetic data" and we would have built
an overfitting machine with a validation ritual bolted on. Training a separate
generator per fold costs n_folds times as much compute and is the only way the
selected ratio means what it claims to mean.

The E (evaluation) session is never loaded by this script at all.

Outputs, per subject, in Code/GAN/checkpoints/:
    A0x_gan.pt     generator weights (all folds) + training history + config
    A0x_synth.npz  pre-sampled synthetic trials in microvolts, plus the fold
                   assignment main.py must reuse

Usage
-----
    python Code/GAN/train_gan.py                      # all 9 subjects
    python Code/GAN/train_gan.py --subjects A01 A03   # a couple
    python Code/GAN/train_gan.py --quick              # tiny budget, smoke test
    python Code/GAN/train_gan.py --no-gate-folds      # full generator only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))

import data as D
from diffaug import DEFAULT_POLICY
from eeg_wgan import train_gan, sample

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


def _synth_pool_size(n_train, multiple):
    """How many synthetic trials to pre-sample, per fold.

    We cache a pool `multiple` times the size of the real training set so
    main.py can try every augmentation ratio up to that multiple without ever
    re-invoking the generator - which keeps main.py free of a torch
    dependency and makes its runs exactly reproducible from the cache.
    Rounded up to an even number so the two classes stay balanced.
    """
    n = int(np.ceil(n_train * multiple))
    return n + (n % 2)


def train_one(X_real_raw, y_real, args, tag, seed):
    """
    Fit one generator on the given raw trials and return its synthetic pool.

    X_real_raw : (trials, 875, 22) raw microvolts. Normalisation statistics are
                 fitted here, on these trials only - which is what makes the
                 per-fold generators genuinely blind to their held-out data.
    """
    Z, stats = D.to_gan_space(X_real_raw)
    X_cf = D.channels_first(Z)

    t0 = time.time()
    G, history = train_gan(
        X_cf, y_real,
        steps_per_stage=args.steps,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        lr=args.lr,
        n_critic=args.n_critic,
        up_mode=args.up_mode,
        aug_policy="" if args.no_diffaug else DEFAULT_POLICY,
        device=args.device,
        seed=seed,
        verbose=not args.quiet,
    )
    elapsed = time.time() - t0

    n_pool = _synth_pool_size(len(X_real_raw), args.pool_multiple)
    Z_fake, y_fake = sample(
        G, n_per_class=n_pool // 2, n_samples=D.BASE_SAMPLES,
        device=args.device, seed=seed + 7777,
    )
    # Back to (trials, samples, channels) and back to microvolts, using the
    # SAME statistics the generator was trained under.
    X_fake = D.from_gan_space(D.channels_last(Z_fake), stats)

    print(f"    [{tag}] {len(X_real_raw)} real -> {len(X_fake)} synthetic "
          f"in {elapsed/60:.1f} min", flush=True)
    return G, history, X_fake, y_fake


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+", default=D.SUBJECTS)
    p.add_argument("--steps", type=int, default=1200,
                   help="generator steps per resolution stage (6 stages)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-critic", type=int, default=5)
    p.add_argument("--up-mode", default="cubic", choices=["cubic", "linear", "nearest"],
                   help="'nearest' is provided only to reproduce Hartmann's "
                        "aliasing comparison; do not use it for real runs")
    p.add_argument("--no-diffaug", action="store_true",
                   help="disable adaptive differentiable augmentation "
                        "(ablation only - expect memorisation)")
    p.add_argument("--gate-folds", type=int, default=3,
                   help="folds for the leak-free selection gate in main.py")
    p.add_argument("--no-gate-folds", action="store_true",
                   help="train only the full generator; main.py will then have "
                        "to fall back to a fixed ratio")
    p.add_argument("--pool-multiple", type=float, default=2.0,
                   help="cache this many synthetic trials per real trial")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="60 steps/stage, 1 gate fold - shape check only, the "
                        "samples are noise")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.steps = 60
        args.gate_folds = 1
        args.pool_multiple = 1.0

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={args.device}  steps/stage={args.steps}  "
          f"gate_folds={0 if args.no_gate_folds else args.gate_folds}")

    for subject in args.subjects:
        print(f"\n{'='*70}\n{subject}\n{'='*70}", flush=True)

        # Only the T session. The E session is not touched anywhere here.
        X_train, y_train, _, _ = D.load_subject(subject)
        print(f"  {len(X_train)} training trials "
              f"(left={int((y_train==1).sum())}, right={int((y_train==2).sum())})")

        store = {}
        weights = {}

        G, hist, Xf, yf = train_one(X_train, y_train, args, "full",
                                    args.seed + hash(subject) % 1000)
        store["X_full"], store["y_full"] = Xf, yf
        weights["full"] = G.state_dict()
        histories = {"full": hist}

        if not args.no_gate_folds:
            fold_ids = D.gate_fold_ids(y_train, args.gate_folds)
            store["fold_ids"] = fold_ids
            for k in range(args.gate_folds):
                tr = fold_ids != k          # generator sees these
                # and is blind to fold_ids == k, which is what the gate scores.
                Gk, hk, Xk, yk = train_one(
                    X_train[tr], y_train[tr], args, f"fold{k}",
                    args.seed + 100 * (k + 1) + hash(subject) % 1000,
                )
                store[f"X_fold{k}"], store[f"y_fold{k}"] = Xk, yk
                weights[f"fold{k}"] = Gk.state_dict()
                histories[f"fold{k}"] = hk
        else:
            store["fold_ids"] = np.full(len(y_train), -1)

        store["config"] = np.array(json.dumps({
            "subject": subject,
            "steps_per_stage": args.steps,
            "gate_folds": 0 if args.no_gate_folds else args.gate_folds,
            "up_mode": args.up_mode,
            "diffaug": not args.no_diffaug,
            "base_window": list(D.BASE_WINDOW),
            "n_train": int(len(X_train)),
        }))

        np.savez_compressed(CKPT_DIR / f"{subject}_synth.npz", **store)
        torch.save({"weights": weights, "history": histories, "args": vars(args)},
                   CKPT_DIR / f"{subject}_gan.pt")
        print(f"  saved -> {CKPT_DIR / (subject + '_synth.npz')}")

    print("\nDone. Next: python Code/GAN/evaluate_gan.py   then   "
          "python Code/GAN/main.py")


if __name__ == "__main__":
    main()
