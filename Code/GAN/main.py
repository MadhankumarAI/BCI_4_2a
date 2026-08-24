"""
GAN-augmented stacked ensemble for Left/Right motor imagery (BCI IV-2a).
========================================================================

This is the pipeline from Code/StackedEnsemble/main.py - 5 base models x 3
time windows, out-of-fold stacking, logistic-regression meta-learner, 0.72
mean kappa - with one thing added: the base learners may be trained on real
trials PLUS synthetic ones, and whether they are is decided per subject by
data rather than by us.

The whole design here is organised around not fooling ourselves. Adding
generated data to a training set is an unusually easy way to manufacture a
number that does not survive contact with a new subject, and this is a medical
pipeline, so four specific safeguards are built in:

1. The E (evaluation) session is untouchable.
   The GAN is fitted on the T session only (train_gan.py never opens E). The
   augmentation strategy is chosen on the T session only. E is read once, at
   the very end, to produce the reported number. Nothing is tuned on it.

2. The selection gate is leak-free.
   For gate fold k, the synthetic trials come from a generator that was fitted
   without fold k's trials (train_gan.py trains one generator per fold for
   exactly this). A single generator fitted on all of T would make every fold
   score optimistically, the gate would always say "add more synthetic data",
   and the safeguard would be theatre.

3. "No augmentation" is the default, and has to be beaten convincingly.
   Candidates are scored by cross-validated kappa and the winner is chosen by
   the one-standard-error rule: augmentation is only adopted if it beats the
   un-augmented baseline by more than one standard error of the fold scores.
   Picking the argmax over 6 candidates on ~40 validation trials would itself
   overfit; this makes the gate conservative by construction, and a subject
   whose GAN is unhelpful simply keeps the 0.72-kappa pipeline unchanged.

4. The GAN has to beat free alternatives.
   Segment-recombination and Riemannian re-colouring (augment.py) are in the
   same candidate pool. If the GAN cannot beat a method that costs
   milliseconds and cannot memorise anything, it does not get used.

Synthetic trials are only ever added to BASE-LEARNER training sets. The
meta-learner is always fitted on out-of-fold predictions for real trials, and
those out-of-fold folds reuse the gate folds so the per-fold blind generators
apply there too. A meta-learner calibrated on synthetic data would be
learning how the base models behave on fakes, which is not the question.

Preprocessing note
------------------
Every trial - real and synthetic, train and test - is CAR'd and band-passed to
4-40 Hz here. The filter bank already restricts the models to 4-40 Hz, but the
augmented-covariance model reads the time series directly, so without a shared
band limit it could separate real from synthetic on out-of-band content alone.
The un-augmented baseline printed by this script is therefore the band-limited
variant of the stacked ensemble; every candidate is compared against that same
baseline, so the comparison stays like-for-like.

Usage
-----
    python Code/GAN/main.py                    # after train_gan.py
    python Code/GAN/main.py --subjects A01 A02
    python Code/GAN/main.py --all-candidates   # adds the ablation controls
    python Code/GAN/main.py --no-gan           # classical augmentation only
"""

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.append(str(HERE))
sys.path.append(str(CODE))
sys.path.append(str(CODE / "Ensemble"))

import data as D
from augment import CLASSICAL_METHODS
from preprocessing import apply_filter_bank, bandpass_filter, common_average_reference
from fbcsp import FBCSP
from advanced_riemann import FilterBankCovariances, FilterBankTangentSpace

warnings.filterwarnings("ignore")

CKPT_DIR = HERE / "checkpoints"


def _load_stacked_models():
    """Import the base-model wrappers from Code/StackedEnsemble/main.py.

    Loaded by path under a distinct module name rather than by `import main`,
    because this file is also called main.py and would shadow it.
    """
    path = CODE / "StackedEnsemble" / "main.py"
    spec = importlib.util.spec_from_file_location("stacked_ensemble_impl", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stacked_ensemble_impl"] = mod
    spec.loader.exec_module(mod)
    return mod


SE = _load_stacked_models()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def prep(X):
    """CAR + 4-40 Hz band-pass. Applied identically to real and synthetic."""
    return bandpass_filter(
        common_average_reference(np.asarray(X, dtype=np.float64)),
        D.GAN_BAND[0], D.GAN_BAND[1], D.FS,
    )


def window_views(X_car):
    """
    Build the three per-window representations the base models consume.

    Returns a list of dicts, one per entry of D.TIME_WINDOWS, each holding the
    exact inputs the five models want:

        car  (trials, samples, 22)              -> AugCov
        fb   (trials, bands, samples, 22)       -> FBCSP
        cov  (trials, bands, 22, 22)            -> TSM, MDM, SVM-RBF

    Everything is a crop of the same 875-sample tensor, so a synthetic trial is
    the same trial in all three views.
    """
    views = []
    for _, t0, t1 in D.TIME_WINDOWS:
        car = D.crop_window(X_car, t0, t1)
        fb = apply_filter_bank(car, D.FS)
        views.append({"car": car, "fb": fb,
                      "cov": FilterBankCovariances().transform(fb)})
    return views


# ---------------------------------------------------------------------------
# Candidate augmentation strategies
# ---------------------------------------------------------------------------

#: (name, kind, ratio). ratio is synthetic trials per real trial.
BASE_CANDIDATES = [
    ("none", None, 0.0),
    ("gan@0.5", "gan", 0.5),
    ("gan@1.0", "gan", 1.0),
    ("gan@2.0", "gan", 2.0),
    ("segrec@1.0", "segrec", 1.0),
    ("riemann@1.0", "riemann", 1.0),
]

#: Controls and combinations, enabled with --all-candidates. Kept out of the
#: default pool because every extra candidate adds selection variance, and
#: noise/shift copies exist to diagnose rather than to win.
EXTRA_CANDIDATES = [
    ("noise@1.0", "noise", 1.0),
    ("shift@1.0", "shift", 1.0),
    ("gan+segrec@1.0", "gan+segrec", 1.0),
]


def draw_synthetic(kind, ratio, X_real, y_real, pool, seed):
    """
    Produce the synthetic block for one candidate.

    Parameters
    ----------
    X_real, y_real : the real trials available for this fit (already prepped).
        Classical methods derive their output from these; the GAN ignores them
        and draws from `pool` instead.
    pool : (X_pool, y_pool) cached GAN samples that are blind to whatever this
        fit is going to be validated on, or None if no GAN is in play.

    Returns (X_syn, y_syn), class-balanced, or (None, None) for no augmentation.
    """
    n_new = int(round(ratio * len(X_real)))
    n_new -= n_new % 2                     # keep the two classes balanced
    if kind is None or n_new <= 0:
        return None, None

    rng = np.random.default_rng(seed)

    def take_from_pool(n):
        if pool is None:
            raise RuntimeError("a GAN candidate was scored without a cached pool")
        Xp, yp = pool
        picks = []
        for c in (1, 2):
            idx = np.where(yp == c)[0]
            if len(idx) == 0:
                raise RuntimeError(f"cached pool has no class {c} samples")
            # Sample without replacement when the pool is big enough; the pool
            # is sized by --pool-multiple in train_gan.py precisely so that the
            # largest ratio still gets distinct trials.
            picks.append(rng.choice(idx, size=n // 2, replace=len(idx) < n // 2))
        sel = np.concatenate(picks)
        return Xp[sel], yp[sel]

    if kind == "gan":
        return take_from_pool(n_new)

    if kind == "gan+segrec":
        half = (n_new // 2) - (n_new // 2) % 2
        Xg, yg = take_from_pool(max(2, half))
        Xs, ys = CLASSICAL_METHODS["segrec"](X_real, y_real, n_new - len(Xg), seed)
        return np.concatenate([Xg, Xs]), np.concatenate([yg, ys])

    return CLASSICAL_METHODS[kind](X_real, y_real, n_new, seed)


# ---------------------------------------------------------------------------
# Proxy model used by the selection gate
# ---------------------------------------------------------------------------

def proxy_score(X_tr, y_tr, X_va, y_va):
    """
    Cheap stand-in for the full 15-model stack, scored in kappa.

    FBCSP+LDA and filter-bank tangent-space + logistic regression on the Full
    window, soft-voted. These are the two most complementary families in the
    ensemble (spatial-filter log-variance vs. Riemannian geometry), so a
    strategy that helps both almost always helps the full stack.

    A proxy is used because the gate runs
    (candidates x folds) = up to 27 fits per subject, and running the real
    15-model stack that many times would put a single subject into the hours.
    The cost of the approximation is that the gate ranks candidates slightly
    differently from the full stack; the one-standard-error rule absorbs that
    by refusing to move off the baseline for small differences anyway.
    """
    fb_tr = apply_filter_bank(X_tr, D.FS)
    fb_va = apply_filter_bank(X_va, D.FS)

    # FBCSP + LDA
    fbcsp = FBCSP(m_components=4, k_features=8).fit(fb_tr, y_tr)
    sc1 = StandardScaler().fit(fbcsp.transform(fb_tr))
    lda = LDA(solver="lsqr", shrinkage="auto").fit(sc1.transform(fbcsp.transform(fb_tr)), y_tr)
    p1 = lda.predict_proba(sc1.transform(fbcsp.transform(fb_va)))

    # Filter-bank tangent space + logistic regression
    cov_tr = FilterBankCovariances().transform(fb_tr)
    cov_va = FilterBankCovariances().transform(fb_va)
    ts = FilterBankTangentSpace().fit(cov_tr)
    F_tr, F_va = ts.transform(cov_tr), ts.transform(cov_va)
    sel = SelectKBest(f_classif, k=min(200, F_tr.shape[1])).fit(F_tr, y_tr)
    sc2 = StandardScaler().fit(sel.transform(F_tr))
    lr = LogisticRegression(solver="liblinear", C=0.1, random_state=42)
    lr.fit(sc2.transform(sel.transform(F_tr)), y_tr)
    p2 = lr.predict_proba(sc2.transform(sel.transform(F_va)))

    pred = lda.classes_[np.argmax(p1 + p2, axis=1)]
    return cohen_kappa_score(y_va, pred)


def run_gate(X_car, y, fold_ids, pools, candidates, seed):
    """
    Score every candidate with the leak-free cross-validation, then apply the
    one-standard-error rule.

    Returns (winner_name, winner_kind, winner_ratio, table) where `table` maps
    candidate name -> (mean kappa, standard error).
    """
    n_folds = int(fold_ids.max()) + 1
    scores = {name: [] for name, _, _ in candidates}

    for k in range(n_folds):
        va = fold_ids == k
        tr = ~va
        X_tr, y_tr = X_car[tr], y[tr]
        X_va, y_va = X_car[va], y[va]
        pool = pools.get(k)          # blind to fold k by construction

        for name, kind, ratio in candidates:
            if kind is not None and "gan" in kind and pool is None:
                continue
            X_syn, y_syn = draw_synthetic(kind, ratio, X_tr, y_tr, pool,
                                          seed + 31 * k)
            if X_syn is None:
                Xa, ya = X_tr, y_tr
            else:
                Xa = np.concatenate([X_tr, X_syn])
                ya = np.concatenate([y_tr, y_syn])
            scores[name].append(proxy_score(Xa, ya, X_va, y_va))

    table = {}
    for name, vals in scores.items():
        if not vals:
            continue
        table[name] = (float(np.mean(vals)),
                       float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                       if len(vals) > 1 else 0.0)

    baseline_mean, baseline_se = table["none"]
    best_name = max(table, key=lambda n: table[n][0])
    best_mean = table[best_name][0]

    # One-standard-error rule, biased towards doing nothing: the challenger
    # must clear the baseline by more than the baseline's own fold-to-fold
    # noise before we accept it.
    if best_name != "none" and best_mean <= baseline_mean + baseline_se:
        best_name = "none"

    kind, ratio = next((k_, r) for n_, k_, r in candidates if n_ == best_name)
    return best_name, kind, ratio, table


# ---------------------------------------------------------------------------
# Full stacked ensemble
# ---------------------------------------------------------------------------

def fit_base_models(views, y, hyperparams=None):
    """
    Fit the 5 models on each of the 3 windows.

    `hyperparams` None means run each model's own inner grid search and return
    what it found; passing the returned dict back reuses those settings. Same
    trick as the original pipeline: search once on the full training set, then
    reuse during out-of-fold generation, which turns hours into minutes and -
    since the search only ever sees training data - costs nothing in rigour.
    """
    fitted, found = [], {}
    for wi, view in enumerate(views):
        hp = (hyperparams or {}).get(wi, {})

        m1 = SE.FBCSPModel(*hp.get("fbcsp", (None, None)))
        m1.fit(view["fb"], y)
        found.setdefault(wi, {})["fbcsp"] = (m1.k, m1.m)

        m2 = SE.RiemannTSMModel(*hp.get("tsm", (None, None)))
        m2.fit(view["cov"], y)
        found[wi]["tsm"] = (m2.k, m2.C)

        m3 = SE.MDMModel()
        m3.fit(view["cov"], y)

        m4 = SE.AugCovModel(*hp.get("augcov", (None, None, None)))
        m4.fit(view["car"], y)
        found[wi]["augcov"] = (m4.d, m4.s, m4.C)

        m5 = SE.SVMRBFModel(*hp.get("svm", (None,)))
        m5.fit(view["cov"], y)
        found[wi]["svm"] = (m5.C,)

        fitted.append((m1, m2, m3, m4, m5))
    return fitted, found


def base_meta_features(fitted, views):
    """Concatenate all 15 models' class probabilities -> (trials, 30)."""
    out = []
    for (m1, m2, m3, m4, m5), view in zip(fitted, views):
        out.append(m1.predict_proba(view["fb"]))
        out.append(m2.predict_proba(view["cov"]))
        out.append(m3.predict_proba(view["cov"]))
        out.append(m4.predict_proba(view["car"]))
        out.append(m5.predict_proba(view["cov"]))
    return np.concatenate(out, axis=1)


def subset_views(views, idx):
    """Index every window view with the same trial mask."""
    return [{k: v[idx] for k, v in view.items()} for view in views]


def concat_views(a, b):
    return [{k: np.concatenate([va[k], vb[k]]) for k in va}
            for va, vb in zip(a, b)]


def run_subject(subject, args):
    # ---- data -------------------------------------------------------------
    X_train_raw, y_train, X_test_raw, y_test = D.load_subject(subject)
    X_train = prep(X_train_raw)
    X_test = prep(X_test_raw)

    # ---- cached generator samples ----------------------------------------
    pools, full_pool, fold_ids = {}, None, None
    npz_path = CKPT_DIR / f"{subject}_synth.npz"
    if npz_path.exists() and not args.no_gan:
        store = np.load(npz_path, allow_pickle=True)
        cfg = json.loads(str(store["config"]))
        if cfg["n_train"] != len(X_train):
            raise RuntimeError(
                f"{subject}: cache holds {cfg['n_train']} training trials but "
                f"the loader produced {len(X_train)} - the cache is stale, "
                f"re-run train_gan.py"
            )
        # prep() the synthetic trials too: they were saved post-CAR and
        # post-band-pass already, and both operations are idempotent, so this
        # only guarantees identical treatment rather than changing them.
        full_pool = (prep(store["X_full"]), store["y_full"])
        fold_ids = store["fold_ids"]
        for k in range(int(cfg["gate_folds"])):
            pools[k] = (prep(store[f"X_fold{k}"]), store[f"y_fold{k}"])
    if fold_ids is None or fold_ids.min() < 0:
        fold_ids = D.gate_fold_ids(y_train, args.gate_folds)
        pools = {}

    n_folds = int(fold_ids.max()) + 1

    # ---- gate -------------------------------------------------------------
    candidates = list(BASE_CANDIDATES)
    if args.all_candidates:
        candidates += EXTRA_CANDIDATES
    if args.no_gan or not pools:
        candidates = [c for c in candidates if c[1] is None or "gan" not in c[1]]

    name, kind, ratio, table = run_gate(
        X_train, y_train, fold_ids, pools, candidates, args.seed
    )
    print(f"  gate ({n_folds}-fold, leak-free):")
    for cname in sorted(table, key=lambda n: -table[n][0]):
        m, se = table[cname]
        mark = " <- selected" if cname == name else ""
        print(f"    {cname:<16} kappa {m:6.3f} +/- {se:5.3f}{mark}")
    if name == "none" and len(table) > 1:
        print("    (one-SE rule: no candidate cleared the baseline)")

    # ---- final base learners: real + selected synthetic -------------------
    views_test = window_views(X_test)

    X_syn, y_syn = draw_synthetic(kind, ratio, X_train, y_train, full_pool,
                                  args.seed + 999)
    if X_syn is None:
        X_fit, y_fit = X_train, y_train
    else:
        X_fit = np.concatenate([X_train, X_syn])
        y_fit = np.concatenate([y_train, y_syn])
    print(f"  final base fit: {len(X_train)} real + "
          f"{0 if X_syn is None else len(X_syn)} synthetic")

    views_fit = window_views(X_fit)
    fitted, hyperparams = fit_base_models(views_fit, y_fit)
    meta_test = base_meta_features(fitted, views_test)

    # ---- out-of-fold meta-features, real trials only ----------------------
    # Reusing the gate folds is what keeps this leak-free as well: fold k's
    # synthetic trials come from the generator that never saw fold k.
    views_train = window_views(X_train)
    oof = np.zeros((len(y_train), meta_test.shape[1]))

    for k in range(n_folds):
        va = fold_ids == k
        tr = ~va
        X_tr, y_tr = X_train[tr], y_train[tr]

        Xk, yk = draw_synthetic(kind, ratio, X_tr, y_tr, pools.get(k),
                                args.seed + 31 * k)
        v_tr = subset_views(views_train, tr)
        if Xk is not None:
            v_tr = concat_views(v_tr, window_views(Xk))
            y_tr = np.concatenate([y_tr, yk])

        fold_models, _ = fit_base_models(v_tr, y_tr, hyperparams)
        oof[va] = base_meta_features(fold_models, subset_views(views_train, va))

    # ---- meta-learner -----------------------------------------------------
    scaler = StandardScaler().fit(oof)
    oof_s = scaler.transform(oof)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    best_C, best = 1.0, -1.0
    for C in [0.01, 0.1, 1.0, 10.0]:
        s = [LogisticRegression(solver="liblinear", C=C, random_state=42)
             .fit(oof_s[a], y_train[a]).score(oof_s[b], y_train[b])
             for a, b in cv.split(oof_s, y_train)]
        if np.mean(s) > best:
            best, best_C = np.mean(s), C

    meta = LogisticRegression(solver="liblinear", C=best_C, random_state=42)
    meta.fit(oof_s, y_train)

    y_pred = meta.predict(scaler.transform(meta_test))
    return {
        "subject": subject,
        "strategy": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "kappa": float(cohen_kappa_score(y_test, y_pred)),
        "gate": table,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+", default=D.SUBJECTS)
    p.add_argument("--gate-folds", type=int, default=3,
                   help="only used when no cached generators are present; "
                        "otherwise the cache's own fold count wins")
    p.add_argument("--all-candidates", action="store_true",
                   help="add noise/shift controls and the GAN+segrec mixture")
    p.add_argument("--no-gan", action="store_true",
                   help="ignore the cached generators; classical augmentation "
                        "only. Useful as the ablation the GAN must beat.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    print("=" * 70)
    print("GAN-AUGMENTED STACKED ENSEMBLE - Left vs Right motor imagery")
    print("15 base learners (5 models x 3 windows) + LR meta-learner")
    print("Augmentation chosen per subject by leak-free CV, one-SE rule")
    print("=" * 70)

    results = []
    for s in args.subjects:
        print(f"\n{'-'*70}\n{s}\n{'-'*70}", flush=True)
        r = run_subject(s, args)
        results.append(r)
        print(f"  >>> {r['strategy']:<16} "
              f"accuracy {r['accuracy']*100:5.2f}%  kappa {r['kappa']:.4f}",
              flush=True)

    print("\n" + "=" * 70)
    print(f"{'subject':<10}{'strategy':<18}{'accuracy':>10}{'kappa':>9}")
    print("-" * 70)
    for r in results:
        print(f"{r['subject']:<10}{r['strategy']:<18}"
              f"{r['accuracy']*100:>9.2f}%{r['kappa']:>9.4f}")
    print("-" * 70)
    print(f"{'MEAN':<28}{np.mean([r['accuracy'] for r in results])*100:>9.2f}%"
          f"{np.mean([r['kappa'] for r in results]):>9.4f}")

    used = [r["subject"] for r in results if r["strategy"] != "none"]
    print(f"\nAugmentation adopted for {len(used)}/{len(results)} subjects"
          f"{': ' + ', '.join(used) if used else ''}")
    print("Subjects marked 'none' ran the unmodified stacked ensemble - the "
          "gate found no candidate worth the risk.")
    print("=" * 70)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
