"""
GAN-ready data layer for BCI Competition IV-2a (Left vs Right motor imagery).
==============================================================================

Everything downstream of this module assumes ONE canonical trial tensor per
subject:

    X : (trials, 875, 22)   float64,  2.5s - 6.0s of the trial
                            (= 0.5s - 4.0s post-cue, cue is at t = 2.0s)

The three windows used by the stacked ensemble are exact sub-crops of it:

    Early (2.5-4.5s) -> X[:,   0:500]
    Full  (2.5-6.0s) -> X[:,   0:875]
    Late  (3.5-6.0s) -> X[:, 250:875]

That is the reason a single generator per subject is enough: we synthesise the
widest window once and crop, so a synthetic trial stays temporally coherent
across all three views the ensemble looks at. Generating each window with its
own GAN would let the three views of the "same" fake trial contradict each
other, which is exactly the kind of artefact a meta-learner will happily latch
onto.

Leakage policy
--------------
`load_subject` returns train (T session) and test (E session) separately and
NOTHING here ever fits a statistic on the test session. Channel means/stds,
the GAN, and the augmentation gate are all train-only. The E session is only
ever transformed with train-derived numbers.

Suemitsu & Nambu (2023) caveat
------------------------------
That paper shows classifiers on IV-2a gain accuracy from the visual-cue and
rest periods rather than from motor imagery, which inflates published numbers.
The windows above start 0.5s AFTER cue onset, which is the standard MI window
in the FBCSP/Riemannian literature, but the 2a cue arrow stays on screen until
3.25s. `STRICT_MI_WINDOW` below is the cue-free alternative (1.25s-4.0s
post-cue) if you want the conservative protocol for a clinical claim.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_loader import load_and_epoch_data, load_true_labels
from preprocessing import bandpass_filter, common_average_reference

FS = 250

# Canonical generation window, in absolute trial seconds (cue at 2.0s).
BASE_WINDOW = (2.5, 6.0)
BASE_SAMPLES = int(round((BASE_WINDOW[1] - BASE_WINDOW[0]) * FS))  # 875

# Windows the stacked ensemble consumes, as (name, start_sec, end_sec).
TIME_WINDOWS = [
    ("early", 2.5, 4.5),
    ("full", 2.5, 6.0),
    ("late", 3.5, 6.0),
]

# Cue-free window (arrow disappears at 3.25s) for the conservative protocol.
STRICT_MI_WINDOW = (3.25, 6.0)

# Broadband limits. The downstream filter bank only ever looks at 4-40 Hz, so
# the GAN is asked to model exactly that band and nothing else: no DC drift, no
# 50 Hz line noise, no EOG residual to waste capacity on.
GAN_BAND = (4.0, 40.0)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "BCICIV-2a-mat"
LABELS_DIR = REPO_ROOT / "true_labels"
SUBJECTS = [f"A0{i}" for i in range(1, 10)]


def crop_window(X, start_sec, end_sec):
    """Crop the canonical (trials, 875, 22) tensor to one ensemble window."""
    a = int(round((start_sec - BASE_WINDOW[0]) * FS))
    b = int(round((end_sec - BASE_WINDOW[0]) * FS))
    if a < 0 or b > X.shape[1]:
        raise ValueError(
            f"window ({start_sec}, {end_sec}) is not inside base {BASE_WINDOW}"
        )
    return X[:, a:b]


def load_subject(subject, window=BASE_WINDOW):
    """
    Load one subject's Left/Right trials for both sessions.

    Returns
    -------
    X_train : (n_train, samples, 22)  raw uV, artefact trials already dropped
    y_train : (n_train,)              labels in {1, 2}
    X_test  : (n_test, samples, 22)
    y_test  : (n_test,)
    """
    t_file = DATA_DIR / f"{subject}T.mat"
    e_file = DATA_DIR / f"{subject}E.mat"
    lbl_file = LABELS_DIR / f"{subject}E.mat"

    X_train, y_train, _ = load_and_epoch_data(
        t_file, start_sec=window[0], end_sec=window[1], fs=FS
    )

    X_test, _, test_idx = load_and_epoch_data(
        e_file, start_sec=window[0], end_sec=window[1], fs=FS
    )
    y_test_all = load_true_labels(lbl_file)[test_idx]
    keep = (y_test_all == 1) | (y_test_all == 2)

    return X_train, y_train, X_test[keep], y_test_all[keep]


def to_gan_space(X, stats=None):
    """
    Map raw trials into the space the GAN actually models.

    CAR -> 4-40 Hz band-pass -> per-channel z-score.

    CAR is a projection (it is idempotent), so synthetic trials produced here
    can be injected straight into the classifier pipeline at the post-CAR stage
    and re-running CAR on them changes nothing.

    Parameters
    ----------
    stats : (mu, sd) or None
        Pass None on the training set to fit the statistics; pass the returned
        tuple everywhere else. Never fit these on the evaluation session.

    Returns
    -------
    Z     : (trials, samples, 22) normalised
    stats : (mu, sd), each (1, 1, 22)
    """
    Xc = common_average_reference(np.asarray(X, dtype=np.float64))
    Xb = bandpass_filter(Xc, GAN_BAND[0], GAN_BAND[1], FS)

    if stats is None:
        mu = Xb.mean(axis=(0, 1), keepdims=True)
        sd = Xb.std(axis=(0, 1), keepdims=True)
        sd = np.maximum(sd, 1e-8)
        stats = (mu, sd)

    mu, sd = stats
    return (Xb - mu) / sd, stats


def from_gan_space(Z, stats):
    """Undo the z-score so synthetic trials are back in microvolt scale."""
    mu, sd = stats
    return Z * sd + mu


GATE_FOLD_SEED = 1234


def gate_fold_ids(y, n_folds, seed=GATE_FOLD_SEED):
    """
    Stratified fold assignment for the augmentation-selection gate.

    train_gan.py and main.py MUST agree on this split exactly: the whole point
    of the gate is that fold k's synthetic trials come from a GAN that never
    saw fold k's validation trials. If the two scripts disagree about which
    trials are in fold k, the guarantee silently evaporates and the gate
    becomes optimistic. Hence one function, one fixed seed, imported by both.

    Returns
    -------
    (n_trials,) int array of fold indices in [0, n_folds).
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    ids = np.empty(len(y), dtype=int)
    for k, (_, val) in enumerate(skf.split(np.zeros(len(y)), y)):
        ids[val] = k
    return ids


def channels_first(X):
    """(trials, samples, channels) -> (trials, channels, samples) for torch."""
    return np.ascontiguousarray(np.swapaxes(X, 1, 2))


def channels_last(X):
    """(trials, channels, samples) -> (trials, samples, channels)."""
    return np.ascontiguousarray(np.swapaxes(X, 1, 2))
