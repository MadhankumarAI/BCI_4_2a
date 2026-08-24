"""
Differentiable augmentation for small-sample EEG GAN training.
==============================================================

With ~110 training trials per subject the critic memorises the real set within
a few hundred steps; from then on it hands the generator a gradient that says
"be one of these 110 signals" and the GAN degenerates into a copy machine.
That failure mode is invisible in the loss curves and lethal for a medical
claim, because the "synthetic" data is then just the training set with noise.

The fix that works at this sample size is Zhao et al. (DiffAugment) / Karras
et al. (ADA): apply the SAME differentiable augmentation to real and fake
batches before they reach the critic. The critic then has to discriminate
distributions-under-augmentation, which it cannot do by memorising, and
because the ops are differentiable the generator still receives clean
gradients. Crucially the augmentation is only ever seen by the critic - the
generator's output distribution is left unbiased, unlike augmenting the
training set directly.

The ops below are chosen to be label-preserving for motor imagery:

  time_shift      circular roll. ERD/ERS timing jitters trial to trial anyway.
  amp_scale       per-trial gain. Electrode impedance drifts within a session.
  channel_scale   per-channel gain. Same, but per electrode.
  channel_drop    zeroes a few electrodes. Mimics a bad contact; forces the
                  critic off any single-electrode shortcut.
  noise           additive Gaussian at a fraction of signal std.
  cutout          zeroes a short time span. Blocks "memorise this exact
                  transient" shortcuts.

Deliberately NOT included: channel permutation and time reversal. Both destroy
the left/right lateralisation (C3 vs C4) that is the entire signal we are
trying to model.
"""

import torch


def rand_time_shift(x, max_frac=0.10):
    """Circularly roll each trial by a random amount, up to +/- max_frac."""
    n, _, t = x.shape
    max_shift = max(1, int(t * max_frac))
    shifts = torch.randint(-max_shift, max_shift + 1, (n,), device=x.device)
    idx = torch.arange(t, device=x.device).unsqueeze(0) - shifts.unsqueeze(1)
    idx = idx % t
    return torch.gather(x, 2, idx.unsqueeze(1).expand(-1, x.shape[1], -1))


def rand_amp_scale(x, lo=0.8, hi=1.2):
    """One random gain per trial."""
    g = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(lo, hi)
    return x * g


def rand_channel_scale(x, lo=0.9, hi=1.1):
    """One random gain per electrode per trial."""
    g = torch.empty(x.shape[0], x.shape[1], 1, device=x.device).uniform_(lo, hi)
    return x * g


def rand_channel_drop(x, p=0.1):
    """Zero out each electrode independently with probability p."""
    mask = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > p).float()
    return x * mask


def rand_noise(x, sigma=0.05):
    """Additive Gaussian noise, sigma relative to the per-trial std."""
    scale = x.std(dim=(1, 2), keepdim=True) * sigma
    return x + torch.randn_like(x) * scale


def rand_cutout(x, max_frac=0.15):
    """Zero one random contiguous time span per trial."""
    n, _, t = x.shape
    length = max(1, int(t * max_frac))
    starts = torch.randint(0, max(1, t - length), (n, 1), device=x.device)
    grid = torch.arange(t, device=x.device).unsqueeze(0)
    mask = ((grid < starts) | (grid >= starts + length)).float().unsqueeze(1)
    return x * mask


_POLICIES = {
    "shift": rand_time_shift,
    "amp": rand_amp_scale,
    "chanscale": rand_channel_scale,
    "chandrop": rand_channel_drop,
    "noise": rand_noise,
    "cutout": rand_cutout,
}

DEFAULT_POLICY = "shift,amp,chanscale,chandrop,noise,cutout"


def diff_augment(x, policy=DEFAULT_POLICY, strength=1.0):
    """
    Apply the augmentation chain to a batch.

    Parameters
    ----------
    x : (batch, channels, samples) tensor
    policy : comma-separated op names, or "" / None to disable
    strength : in [0, 1]; probability that each individual op fires. Used by
        the adaptive controller in eeg_wgan.py, which raises it while the
        critic is overfitting and lowers it when it is not.
    """
    if not policy or strength <= 0.0:
        return x
    for name in policy.split(","):
        name = name.strip()
        if not name:
            continue
        fn = _POLICIES.get(name)
        if fn is None:
            raise KeyError(f"unknown diffaug op {name!r}")
        if strength >= 1.0 or torch.rand(()).item() < strength:
            x = fn(x)
    return x
