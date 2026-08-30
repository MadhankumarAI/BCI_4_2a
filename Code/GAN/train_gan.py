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

Outputs, per subject, in the checkpoint directory configured in paths.py:
    A0x_gan.pt     generator weights (all folds) + training history + config
    A0x_synth.npz  pre-sampled synthetic trials in microvolts, plus the fold
                   assignment main.py must reuse

RUNTIME - read this first
-------------------------
This model is expensive on CPU. The script measures a few real steps at each
resolution and prints an estimate before it starts; check that number against
how long you are willing to wait, and use --preset to trade quality for time:

    --preset smoke      seconds per generator. Output is noise. Plumbing only.
    --preset fast       iteration speed. Usable but under-trained.
    --preset balanced   the default.
    --preset thorough   closest to the Hartmann et al. budget.

--dry-run prints the estimate and exits without training anything.

Usage
-----
    python Code/GAN/train_gan.py --dry-run           # how long will this take
    python Code/GAN/train_gan.py --preset smoke --subjects A01
    python Code/GAN/train_gan.py                     # all 9 subjects
    python Code/GAN/train_gan.py --subjects A01 A03
    python Code/GAN/train_gan.py --no-gate-folds     # full generator only
"""

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))

import paths as P
import data as D
from diffaug import DEFAULT_POLICY
from eeg_wgan import train_gan, sample, estimate_minutes, stage_steps

#: name -> (steps_per_stage, gate_folds, pool_multiple)
PRESETS = {
    "smoke": (8, 2, 1.0),       # seconds; output is noise, plumbing check only
    "fast": (250, 2, 3.0),      # ~4 h for all 9 subjects on a 12-thread CPU
    "balanced": (700, 3, 3.0),  # ~14 h
    "thorough": (1500, 3, 4.0), # ~29 h
}


def _stable_seed(subject, offset=0):
    """Deterministic per-subject seed.

    Python's built-in hash() is randomised per process unless PYTHONHASHSEED
    is set, so using it here would silently make every run irreproducible -
    which for a pipeline whose whole argument is "the protocol is auditable"
    is not a small thing. crc32 is stable across processes and platforms.
    """
    return int(zlib.crc32(subject.encode())) % 100000 + offset


def _synth_pool_size(n_train, multiple):
    """How many synthetic trials to pre-sample, per fold.

    We cache a pool `multiple` times the size of the real training set so
    main.py can try every augmentation ratio up to that multiple without ever
    re-invoking the generator - which keeps main.py free of a torch
    dependency and makes its runs exactly reproducible from the cache. The
    default multiple is above the largest candidate ratio (2.0) so even that
    candidate draws distinct trials rather than duplicates.
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
        gp_every=args.gp_every,
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
    X_raw_fake = D.from_gan_space(D.channels_last(Z_fake), stats)

    # Measure how much of the generator's output the pipeline is about to
    # discard, then discard it - so the cache holds data in the same space as
    # the real trials it will sit beside.
    lost = D.discarded_power(X_raw_fake)
    X_fake = D.postprocess_synthetic(X_raw_fake)

    # Amplitude sanity: the single number that exposes an under-trained
    # generator fastest. Real band-passed trials and synthetic ones should
    # have comparable per-channel spread.
    real_std = float(D.postprocess_synthetic(X_real_raw).std())
    amp = float(X_fake.std()) / max(real_std, 1e-12)

    print(f"    [{tag}] {len(X_real_raw)} real -> {len(X_fake)} synthetic "
          f"in {elapsed/60:.1f} min  |  lost to CAR "
          f"{lost['common_mode']*100:.0f}%, to band-pass "
          f"{lost['out_of_band']*100:.0f}%  |  amplitude x{amp:.2f} of real",
          flush=True)
    if lost["common_mode"] > 0.05:
        print(f"      WARNING {lost['common_mode']*100:.0f}% of generator "
              f"power is common-mode and gets deleted by CAR. The generator "
              f"should be projecting this out - check Generator._finish.",
              flush=True)
    if amp < 0.5 or amp > 2.0:
        print(f"      WARNING amplitude is off by more than 2x - this "
              f"generator is under-trained; raise --steps / --preset",
              flush=True)
    return G, history, X_fake, y_fake


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+", default=P.SUBJECTS)
    p.add_argument("--preset", default="balanced", choices=sorted(PRESETS),
                   help="quality/time trade-off; --steps, --gate-folds and "
                        "--pool-multiple override individual fields")
    p.add_argument("--steps", type=int, default=None,
                   help="generator steps at the base resolution; later stages "
                        "are scaled down by eeg_wgan.STAGE_STEP_SCALE")
    p.add_argument("--batch-size", type=int, default=16,
                   help="16 rather than the usual 32: on CPU this model is "
                        "compute-bound and batch size is close to a linear "
                        "cost, while WGAN-GP with minibatch-stddev is stable "
                        "well below 32")
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-critic", type=int, default=5)
    p.add_argument("--gp-every", type=int, default=4,
                   help="lazy gradient penalty interval; 1 = textbook WGAN-GP")
    p.add_argument("--up-mode", default="cubic", choices=["cubic", "linear", "nearest"],
                   help="'nearest' is provided only to reproduce Hartmann's "
                        "aliasing comparison; do not use it for real runs")
    p.add_argument("--no-diffaug", action="store_true",
                   help="disable adaptive differentiable augmentation "
                        "(ablation only - expect memorisation)")
    p.add_argument("--gate-folds", type=int, default=None,
                   help="folds for the leak-free selection gate in main.py")
    p.add_argument("--no-gate-folds", action="store_true",
                   help="train only the full generator; main.py will then have "
                        "to fall back to a fixed ratio")
    p.add_argument("--pool-multiple", type=float, default=None,
                   help="cache this many synthetic trials per real trial")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="print the time estimate and exit")
    p.add_argument("--skip-existing", action="store_true",
                   help="leave subjects that already have a cache alone")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    # Preset first, explicit flags win.
    steps, folds, pool = PRESETS[args.preset]
    args.steps = args.steps if args.steps is not None else steps
    args.gate_folds = args.gate_folds if args.gate_folds is not None else folds
    args.pool_multiple = (args.pool_multiple if args.pool_multiple is not None
                          else pool)

    P.verify(args.subjects)
    P.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    n_gans_each = 1 + (0 if args.no_gate_folds else args.gate_folds)
    print("Paths:")
    print(P.describe())
    print(f"\npreset={args.preset}  device={args.device}  "
          f"steps/stage(base)={args.steps}  "
          f"per-stage={stage_steps(args.steps)}")
    print(f"gate_folds={0 if args.no_gate_folds else args.gate_folds}  "
          f"-> {n_gans_each} generators x {len(args.subjects)} subjects "
          f"= {n_gans_each * len(args.subjects)} trainings")

    print("\nTiming a few real steps at each resolution...", flush=True)
    per_gan = estimate_minutes(
        args.steps, batch_size=args.batch_size, n_critic=args.n_critic,
        gp_every=args.gp_every, device=args.device,
    )
    total = per_gan * n_gans_each * len(args.subjects)
    print(f"  ~{per_gan:.1f} min per generator")
    print(f"  ~{total/60:.1f} HOURS total for this command")
    if args.dry_run:
        print("\n--dry-run: nothing trained. Drop the flag to start, or pick a "
              "cheaper --preset.")
        return
    if total > 60 and not args.quiet:
        print("  (that is a long run - consider --preset fast, or one "
              "--subjects at a time)", flush=True)

    for subject in args.subjects:
        out_npz = P.synth_file(subject)
        if args.skip_existing and out_npz.exists():
            print(f"\n{subject}: cache exists, skipping")
            continue

        print(f"\n{'='*70}\n{subject}\n{'='*70}", flush=True)

        # Only the T session. The E session is not touched anywhere here.
        X_train, y_train, _, _ = D.load_subject(subject)
        print(f"  {len(X_train)} training trials "
              f"(left={int((y_train==1).sum())}, right={int((y_train==2).sum())})",
              flush=True)

        store = {}
        weights = {}

        G, hist, Xf, yf = train_one(X_train, y_train, args, "full",
                                    _stable_seed(subject, args.seed))
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
                    _stable_seed(subject, args.seed + 100 * (k + 1)),
                )
                store[f"X_fold{k}"], store[f"y_fold{k}"] = Xk, yk
                weights[f"fold{k}"] = Gk.state_dict()
                histories[f"fold{k}"] = hk
        else:
            store["fold_ids"] = np.full(len(y_train), -1)

        store["config"] = np.array(json.dumps({
            "subject": subject,
            "preset": args.preset,
            "steps_per_stage": args.steps,
            "gate_folds": 0 if args.no_gate_folds else args.gate_folds,
            "up_mode": args.up_mode,
            "diffaug": not args.no_diffaug,
            "base_window": list(D.BASE_WINDOW),
            "n_train": int(len(X_train)),
            "n_samples": int(D.BASE_SAMPLES),
        }))

        np.savez_compressed(out_npz, **store)
        torch.save({"weights": weights, "history": histories,
                    "args": vars(args), "n_samples": int(D.BASE_SAMPLES)},
                   P.weights_file(subject))
        print(f"  saved -> {out_npz}", flush=True)

    print("\nDone. Next: python Code/GAN/evaluate_gan.py   then   "
          "python Code/GAN/main.py")


if __name__ == "__main__":
    main()
