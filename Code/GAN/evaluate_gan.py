"""
Fidelity / memorisation audit for the cached generators.
========================================================

Run after train_gan.py, before main.py. Nothing here influences the final
model - main.py makes its own decision from downstream accuracy on held-out
real trials. This script exists to answer, on the record, "is the synthetic
data actually EEG, and is it actually new?", which is the part a reviewer or a
regulator will ask about and which no accuracy number can answer.

Two different comparisons are made, deliberately:

  Fidelity (PSD, band power, Frechet, sliced Wasserstein)
      full generator vs. the full training set it was fitted on. This asks
      "did the generator match the distribution it was given". Comparing a
      model to its own training data is the correct question here - we want
      to know whether it learned the distribution, not whether it generalises.

  Memorisation (nn_distance_ratio) and utility (TSTR / TRTS)
      per gate fold: fold k's generator against the fold-k trials it never
      saw. Asking those two questions against training data would be
      meaningless - a memorising generator would score perfectly on both.

Reading the output
------------------
  nn ratio           want ~1.0. Below ~0.7 means the generator is
                     re-emitting training trials; treat the run as failed.
  tstr               want it within a few points of trtr_baseline. Far below
                     means the synthetic data carries no class information.
  trts               want it near tstr. Much higher means exaggerated,
                     unnaturally separable fakes.
  psd_log_distance   want small. A large value with excess high-frequency
                     power is the upsampling-aliasing failure; re-run with
                     --up-mode cubic.

Usage
-----
    python Code/GAN/evaluate_gan.py
    python Code/GAN/evaluate_gan.py --subjects A01 --json report.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

import paths as P
import data as D
import gan_metrics as M

FIDELITY_KEYS = [
    "amplitude_ratio", "psd_log_distance", "band_err_8-13Hz",
    "band_err_13-30Hz", "frechet_tangent", "sliced_wasserstein",
]


def real_in_gan_space(X_raw):
    """CAR + 4-40 Hz band-pass, back in microvolts.

    The cached synthetic trials went through exactly this, so this is the only
    fair comparison set. Comparing against untouched raw trials would show a
    huge spectral gap that is entirely our own band-pass.
    """
    Z, stats = D.to_gan_space(X_raw)
    return D.from_gan_space(Z, stats)


def evaluate_subject(subject):
    path = P.synth_file(subject)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run train_gan.py for {subject} first"
        )

    store = np.load(path, allow_pickle=True)
    cfg = json.loads(str(store["config"]))

    X_train, y_train, _, _ = D.load_subject(subject)
    X_real = real_in_gan_space(X_train)

    rep = {"subject": subject, "n_train": len(X_train),
           "gate_folds": cfg["gate_folds"], "preset": cfg.get("preset")}
    if cfg.get("preset") == "smoke":
        print(f"NOTE {subject} was trained with --preset smoke; the numbers "
              f"below describe noise, not a real generator.")

    # --- fidelity: full generator vs. its own training distribution ---
    X_fake, y_fake = store["X_full"], store["y_full"]
    rep.update({k: v for k, v in M.full_report(
        X_real, y_train, X_fake, y_fake).items() if k in FIDELITY_KEYS})

    # --- memorisation + utility: per fold, against unseen real trials ---
    fold_ids = store["fold_ids"]
    n_folds = cfg["gate_folds"]
    nn_ratios, tstrs, trtss, baselines = [], [], [], []

    for k in range(n_folds):
        held = fold_ids == k
        if held.sum() < 8:
            continue
        Xk, yk = store[f"X_fold{k}"], store[f"y_fold{k}"]
        X_seen = X_real[~held]

        nn_ratios.append(M.nn_distance_ratio(X_seen, Xk)["ratio"])
        u = M.tstr_trts(X_seen, y_train[~held], Xk, yk,
                        X_real[held], y_train[held])
        tstrs.append(u["tstr"])
        trtss.append(u["trts"])
        baselines.append(u["trtr_baseline"])

    if nn_ratios:
        rep["nn_ratio"] = float(np.mean(nn_ratios))
        rep["tstr"] = float(np.mean(tstrs))
        rep["trts"] = float(np.mean(trtss))
        rep["trtr_baseline"] = float(np.mean(baselines))

    return rep


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+", default=P.SUBJECTS)
    p.add_argument("--json", type=str, default=None,
                   help="also write the full report to this path")
    args = p.parse_args()

    P.verify(args.subjects)

    header = (f"{'subj':<5}{'amp':>7}{'PSD':>8}{'mu err':>9}{'beta err':>10}"
              f"{'FTD':>9}{'SWD':>8}{'nn':>7}{'TSTR':>8}{'TRTS':>8}{'base':>8}")
    print(header)
    print("-" * len(header))

    reports = []
    for s in args.subjects:
        r = evaluate_subject(s)
        reports.append(r)
        print(f"{r['subject']:<5}"
              f"{r['amplitude_ratio']:>7.2f}"
              f"{r['psd_log_distance']:>8.3f}"
              f"{r['band_err_8-13Hz']:>9.3f}"
              f"{r['band_err_13-30Hz']:>10.3f}"
              f"{r['frechet_tangent']:>9.2f}"
              f"{r['sliced_wasserstein']:>8.3f}"
              f"{r.get('nn_ratio', float('nan')):>7.2f}"
              f"{r.get('tstr', float('nan')):>8.3f}"
              f"{r.get('trts', float('nan')):>8.3f}"
              f"{r.get('trtr_baseline', float('nan')):>8.3f}", flush=True)

    def avg(k):
        vals = [r[k] for r in reports if k in r]
        return float(np.mean(vals)) if vals else float("nan")

    print("-" * len(header))
    print(f"{'mean':<5}{avg('amplitude_ratio'):>7.2f}"
          f"{avg('psd_log_distance'):>8.3f}"
          f"{avg('band_err_8-13Hz'):>9.3f}{avg('band_err_13-30Hz'):>10.3f}"
          f"{avg('frechet_tangent'):>9.2f}{avg('sliced_wasserstein'):>8.3f}"
          f"{avg('nn_ratio'):>7.2f}{avg('tstr'):>8.3f}"
          f"{avg('trts'):>8.3f}{avg('trtr_baseline'):>8.3f}")

    weak = [r["subject"] for r in reports
            if not 0.5 <= r["amplitude_ratio"] <= 2.0]
    if weak:
        print(f"\nWARNING amplitude off by more than 2x: {', '.join(weak)}")
        print("These generators are under-trained - their synthetic trials "
              "have the wrong power, so their spatial covariances are wrong "
              "by the square of that. Retrain with a larger --preset before "
              "reading anything else in this table.")

    flagged = [r["subject"] for r in reports if r.get("nn_ratio", 1.0) < 0.7]
    if flagged:
        print(f"\nWARNING memorisation suspected (nn ratio < 0.7): "
              f"{', '.join(flagged)}")
        print("The generator is re-emitting training trials for these "
              "subjects. Do not use their synthetic data; retrain with more "
              "diffaug strength or fewer steps.")

    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
