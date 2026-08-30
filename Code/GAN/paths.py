"""
Every filesystem location the GAN pipeline uses, in one place.
==============================================================

All paths are absolute `pathlib.Path` objects. If you move the project, or run
it on another machine, edit PROJECT_ROOT below and nothing else - every other
path is derived from it.

Each one can also be overridden at run time by an environment variable, which
is what you want on a cluster or in CI where the data sits somewhere else:

    set BCI_PROJECT_ROOT=D:\work\BCI
    set BCI_DATA_DIR=D:\datasets\BCICIV-2a-mat
    set BCI_LABELS_DIR=D:\datasets\true_labels
    set BCI_CHECKPOINT_DIR=D:\scratch\gan_ckpt

(on bash: export BCI_DATA_DIR=/data/BCICIV-2a-mat)

Call `verify()` to check the dataset is actually where these say it is; every
entry-point script does this before it starts work, so a wrong path fails
immediately with a readable message instead of hours later.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Edit here if the project moves.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(
    os.environ.get("BCI_PROJECT_ROOT", r"C:\Users\jaip7\Downloads\madhan\BCI")
)

#: BCI Competition IV Dataset 2a, the .mat files A01T.mat ... A09E.mat
DATA_DIR = Path(os.environ.get("BCI_DATA_DIR", PROJECT_ROOT / "BCICIV-2a-mat"))

#: True labels for the evaluation sessions, A01E.mat ... A09E.mat
LABELS_DIR = Path(os.environ.get("BCI_LABELS_DIR", PROJECT_ROOT / "true_labels"))

#: Where train_gan.py writes generators and cached synthetic trials.
CHECKPOINT_DIR = Path(
    os.environ.get("BCI_CHECKPOINT_DIR", PROJECT_ROOT / "Code" / "GAN" / "checkpoints")
)

#: The existing non-GAN pipelines, imported by Code/GAN/main.py.
CODE_DIR = PROJECT_ROOT / "Code"
STACKED_ENSEMBLE_MAIN = CODE_DIR / "StackedEnsemble" / "main.py"
ENSEMBLE_DIR = CODE_DIR / "Ensemble"

SUBJECTS = [f"A0{i}" for i in range(1, 10)]


def train_file(subject):
    """Absolute path to a subject's T (training) session."""
    return DATA_DIR / f"{subject}T.mat"


def eval_file(subject):
    """Absolute path to a subject's E (evaluation) session."""
    return DATA_DIR / f"{subject}E.mat"


def true_labels_file(subject):
    """Absolute path to a subject's evaluation-session labels."""
    return LABELS_DIR / f"{subject}E.mat"


def synth_file(subject):
    """Cached synthetic trials written by train_gan.py."""
    return CHECKPOINT_DIR / f"{subject}_synth.npz"


def weights_file(subject):
    """Cached generator weights written by train_gan.py."""
    return CHECKPOINT_DIR / f"{subject}_gan.pt"


def verify(subjects=None, need_labels=True):
    """
    Fail fast and loudly if the configured paths are wrong.

    Raises FileNotFoundError listing everything that is missing, rather than
    letting scipy raise a bare "no such file" on the first subject after the
    script has already printed a banner.
    """
    subjects = subjects or SUBJECTS
    problems = []

    if not PROJECT_ROOT.is_dir():
        problems.append(f"PROJECT_ROOT does not exist: {PROJECT_ROOT}")
    if not DATA_DIR.is_dir():
        problems.append(f"DATA_DIR does not exist: {DATA_DIR}")
    if need_labels and not LABELS_DIR.is_dir():
        problems.append(f"LABELS_DIR does not exist: {LABELS_DIR}")

    if not problems:
        for s in subjects:
            for f in (train_file(s), eval_file(s)):
                if not f.is_file():
                    problems.append(f"missing dataset file: {f}")
            if need_labels and not true_labels_file(s).is_file():
                problems.append(f"missing label file: {true_labels_file(s)}")

    if problems:
        raise FileNotFoundError(
            "Dataset paths are not configured correctly.\n  "
            + "\n  ".join(problems)
            + f"\n\nEdit PROJECT_ROOT in {Path(__file__).resolve()}, "
              "or set the BCI_PROJECT_ROOT / BCI_DATA_DIR / BCI_LABELS_DIR "
              "environment variables."
        )


def describe():
    """One-line-per-path summary, printed by the entry-point scripts."""
    return "\n".join([
        f"  PROJECT_ROOT   {PROJECT_ROOT}",
        f"  DATA_DIR       {DATA_DIR}",
        f"  LABELS_DIR     {LABELS_DIR}",
        f"  CHECKPOINT_DIR {CHECKPOINT_DIR}",
    ])


if __name__ == "__main__":
    # `python Code/GAN/paths.py` - check the configuration without running
    # anything expensive.
    print(describe())
    verify()
    print(f"\nAll {len(SUBJECTS)} subjects' dataset and label files are "
          f"present.")
    n_cached = sum(1 for s in SUBJECTS if synth_file(s).exists())
    print(f"Cached generators: {n_cached}/{len(SUBJECTS)} subjects"
          f"{' (run train_gan.py)' if n_cached < len(SUBJECTS) else ''}")
