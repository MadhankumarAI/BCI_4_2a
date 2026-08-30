"""
Conditional progressive WGAN-GP for 22-channel motor-imagery EEG.
=================================================================

Architecture follows Hartmann et al. 2018, "EEG-GAN: Generative adversarial
networks for electroencephalographic brain signals" (arXiv:1806.01875), which
is the reference result for generating *raw* EEG rather than spectrogram
images, plus the modifications that paper found necessary:

  * Progressive growing (Karras et al. 2017). Resolution starts at 28 samples
    and doubles five times to 896. Training the full 3.5 s window from scratch
    on ~110 trials does not converge; growing it does.
  * Interpolation upsampling, never transposed convolution. Section 2.2 of the
    paper measures the aliasing: nearest-neighbour upsampling injects strong
    high-frequency artefacts, linear and cubic inject far weaker ones. Since
    our downstream filter bank reaches 40 Hz, generator-borne aliasing would
    land directly inside the features the classifiers read.
  * Generator block = upsample + 2 convs of kernel 9. Critic block = 2 convs
    + average-pool downsample. LeakyReLU throughout to avoid sparse gradients.
  * Equalised learning rate, pixel normalisation, minibatch standard deviation.
  * n_critic = 5, Adam(lr=1e-3, betas=(0.0, 0.99)), lambda = 10, and a drift
    term 0.001 * E[D(x_r)^2] to keep the critic centred.
  * The paper's own improvement to WGAN-GP: a ONE-SIDED gradient penalty that
    is additionally scaled by the current critic difference, so the constraint
    relaxes as the two distributions converge. Plain WGAN-GP collapsed in
    their table; this did not.

Two things are ours rather than the paper's, both forced by sample size:

  * Class conditioning. Hartmann trained one GAN per class. With 55 trials per
    class that is not viable, so a single conditional model shares all filters
    across left and right and only the class embedding differs. This is the
    projection discriminator of Miyato & Koyama (2018).
  * Adaptive differentiable augmentation (see diffaug.py). The critic
    otherwise memorises the ~110 real trials in a few hundred steps.
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffaug import diff_augment, DEFAULT_POLICY

# torch defaults to physical cores; on this box that leaves half the machine
# idle and CPU training is entirely compute-bound.
torch.set_num_threads(int(os.environ.get("BCI_TORCH_THREADS", os.cpu_count() or 1)))

# Resolution ladder. 896 = 28 * 2**5; the canonical window is 875 samples, so
# the last 21 samples are cropped off after generation.
STAGE_LENGTHS = [28, 56, 112, 224, 448, 896]

# Feature widths per stage. Convolution cost goes as C_in * C_out * length, so
# the top of the ladder dominates: at 896 samples a single block costs more
# than the first three stages together. These widths are deliberately narrow
# at the top - the high-resolution stages are refining detail on structure the
# low-resolution stages already fixed, so they need less capacity than the
# equal-width ladder a GPU implementation would use.
STAGE_CHANNELS = [96, 96, 64, 48, 32, 24]

# Fraction of the step budget spent at each stage, for the same reason. Equal
# budgets (Karras' schedule) spend 60% of wall-clock on the last two stages.
STAGE_STEP_SCALE = [1.0, 1.0, 0.8, 0.6, 0.5, 0.4]

KERNEL = 9


# ---------------------------------------------------------------------------
# Karras-style building blocks
# ---------------------------------------------------------------------------

class EqualConv1d(nn.Module):
    """Conv1d with the He constant applied at runtime instead of at init.

    Equalised learning rate: all weights start at N(0,1) so Adam's per-weight
    second-moment estimates stay on the same scale across layers, and the
    He scaling is folded into the forward pass instead.
    """

    def __init__(self, in_ch, out_ch, kernel, padding=0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.padding = padding
        self.scale = math.sqrt(2.0 / (in_ch * kernel))

    def forward(self, x):
        return F.conv1d(x, self.weight * self.scale, self.bias, padding=self.padding)


class EqualLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.scale = math.sqrt(2.0 / in_f)

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)


class PixelNorm(nn.Module):
    """Normalise each time point across the channel axis.

    Stops the generator from escaping the critic by simply inflating its
    activations, which is the usual precursor to a WGAN blow-up.
    """

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)


class MinibatchStdDev(nn.Module):
    """Append the batch's mean feature std as an extra channel.

    Gives the critic a direct view of within-batch diversity, which is the
    cheapest known deterrent against mode collapse.
    """

    def forward(self, x):
        std = x.std(dim=0, unbiased=False).mean()
        return torch.cat([x, std.expand(x.shape[0], 1, x.shape[2])], dim=1)


def upsample1d(x, mode="cubic"):
    """Double the length by interpolation. Never a transposed convolution."""
    if mode == "cubic":
        # F.interpolate has no 1-D cubic kernel, so borrow the 2-D one on a
        # height-1 image: this applies Catmull-Rom along the time axis only.
        y = F.interpolate(
            x.unsqueeze(2), scale_factor=(1.0, 2.0),
            mode="bicubic", align_corners=False,
        )
        return y.squeeze(2)
    if mode == "linear":
        return F.interpolate(x, scale_factor=2.0, mode="linear", align_corners=False)
    if mode == "nearest":
        return F.interpolate(x, scale_factor=2.0, mode="nearest")
    raise ValueError(f"unknown upsample mode {mode!r}")


def downsample1d(x):
    return F.avg_pool1d(x, 2)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class GenBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up_mode):
        super().__init__()
        self.up_mode = up_mode
        self.c1 = EqualConv1d(in_ch, out_ch, KERNEL, padding=KERNEL // 2)
        self.c2 = EqualConv1d(out_ch, out_ch, KERNEL, padding=KERNEL // 2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = upsample1d(x, self.up_mode)
        x = self.pn(F.leaky_relu(self.c1(x), 0.2))
        x = self.pn(F.leaky_relu(self.c2(x), 0.2))
        return x


class Generator(nn.Module):
    def __init__(self, n_channels=22, latent_dim=128, n_classes=2,
                 class_dim=16, up_mode="cubic"):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.up_mode = up_mode

        self.class_emb = nn.Embedding(n_classes, class_dim)
        nn.init.normal_(self.class_emb.weight, 0.0, 1.0)

        c0 = STAGE_CHANNELS[0]
        self.pn = PixelNorm()
        self.fc = EqualLinear(latent_dim + class_dim, c0 * STAGE_LENGTHS[0])
        self.head = EqualConv1d(c0, c0, KERNEL, padding=KERNEL // 2)

        self.blocks = nn.ModuleList([
            GenBlock(STAGE_CHANNELS[i - 1], STAGE_CHANNELS[i], up_mode)
            for i in range(1, len(STAGE_LENGTHS))
        ])
        # One 1x1 projection to signal space per stage, so a partially grown
        # generator is still a complete model.
        self.to_signal = nn.ModuleList([
            EqualConv1d(c, n_channels, 1) for c in STAGE_CHANNELS
        ])

    def forward(self, z, y, stage, alpha=1.0):
        """
        stage : index into STAGE_LENGTHS; 0 is the 28-sample model.
        alpha : fade-in weight for the newest block. 1.0 = fully faded in.
        """
        h = torch.cat([self.pn(z), self.class_emb(y)], dim=1)
        h = self.fc(h).view(z.shape[0], STAGE_CHANNELS[0], STAGE_LENGTHS[0])
        h = self.pn(F.leaky_relu(h, 0.2))
        h = self.pn(F.leaky_relu(self.head(h), 0.2))

        if stage == 0:
            return self._finish(self.to_signal[0](h))

        for i in range(stage - 1):
            h = self.blocks[i](h)

        prev = h
        h = self.blocks[stage - 1](h)
        out = self.to_signal[stage](h)

        if alpha < 1.0:
            # Blend the new stage against the previous stage's output, simply
            # upsampled. Without this the new block's random init destroys the
            # signal the moment the resolution steps up.
            skip = upsample1d(self.to_signal[stage - 1](prev), self.up_mode)
            out = alpha * out + (1.0 - alpha) * skip
        return self._finish(out)

    @staticmethod
    def _finish(out):
        """
        Project the output onto the subspace the real trials actually occupy.

        The real data satisfies two exact linear constraints before it ever
        reaches the critic, and the generator is free to violate both. Each
        violation costs twice: it hands the critic a trivial real/fake tell
        that has nothing to do with EEG structure, and it spends output range
        on a component the downstream pipeline deletes anyway.

        1. Zero DC per channel (mean over TIME).
           The trials are band-passed to 4-40 Hz, so their per-channel
           temporal mean is zero by construction - measured at 0.003 uV. A
           freshly initialised generator emitted ~0.9 in normalised units,
           as large as the signal itself.

        2. Zero common mode (mean over CHANNELS) - i.e. the CAR constraint.
           Common Average Reference subtracts the across-electrode mean, so
           CAR'd data sums to exactly zero across channels at every time
           point. This one was costing far more than the DC offset: measured
           on a trained generator, re-applying CAR to its output destroyed
           79% of the signal power, because the generator was putting most of
           its energy into a common-mode component that is identically zero in
           every real trial. That showed up downstream as synthetic trials
           with a seventh of the real amplitude.

        Both are mean-removals along different axes, so they commute, and
        removing the channel mean preserves the zero time-mean established
        first. Differentiable and parameter-free.
        """
        out = out - out.mean(dim=2, keepdim=True)   # zero DC per channel
        return out - out.mean(dim=1, keepdim=True)  # zero common mode (CAR)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class CriticBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = EqualConv1d(in_ch, in_ch, KERNEL, padding=KERNEL // 2)
        self.c2 = EqualConv1d(in_ch, out_ch, KERNEL, padding=KERNEL // 2)

    def forward(self, x):
        x = F.leaky_relu(self.c1(x), 0.2)
        x = F.leaky_relu(self.c2(x), 0.2)
        return downsample1d(x)


class Critic(nn.Module):
    def __init__(self, n_channels=22, n_classes=2):
        super().__init__()
        self.from_signal = nn.ModuleList([
            EqualConv1d(n_channels, c, 1) for c in STAGE_CHANNELS
        ])
        self.blocks = nn.ModuleList([
            CriticBlock(STAGE_CHANNELS[i], STAGE_CHANNELS[i - 1])
            for i in range(1, len(STAGE_LENGTHS))
        ])

        c0 = STAGE_CHANNELS[0]
        self.mbstd = MinibatchStdDev()
        self.final_c1 = EqualConv1d(c0 + 1, c0, KERNEL, padding=KERNEL // 2)
        self.final_c2 = EqualConv1d(c0, c0, STAGE_LENGTHS[0])
        self.out = EqualLinear(c0, 1)

        # Projection discriminator (Miyato & Koyama 2018): the class enters as
        # an inner product with the final feature vector rather than as a
        # concatenated input. With 55 trials per class, concatenation lets the
        # critic ignore the label; projection does not.
        self.class_proj = nn.Embedding(n_classes, c0)
        nn.init.zeros_(self.class_proj.weight)

    def forward(self, x, y, stage, alpha=1.0):
        if stage == 0:
            h = F.leaky_relu(self.from_signal[0](x), 0.2)
        else:
            h = F.leaky_relu(self.from_signal[stage](x), 0.2)
            h = self.blocks[stage - 1](h)
            if alpha < 1.0:
                skip = F.leaky_relu(
                    self.from_signal[stage - 1](downsample1d(x)), 0.2
                )
                h = alpha * h + (1.0 - alpha) * skip
            for i in range(stage - 2, -1, -1):
                h = self.blocks[i](h)

        h = self.mbstd(h)
        h = F.leaky_relu(self.final_c1(h), 0.2)
        h = F.leaky_relu(self.final_c2(h), 0.2).flatten(1)

        score = self.out(h).squeeze(1)
        score = score + (self.class_proj(y) * h).sum(dim=1)
        return score


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _relaxed_gradient_penalty(critic, real, fake, y, stage, alpha, w_dist):
    """
    Hartmann et al. section 2.1: one-sided penalty, scaled by the current
    critic difference.

        P1 = E[ max(0, ||grad D(x_hat)||_2 - 1)^2 ]
        term = lambda * max(0, W~) * P1

    The scaling is the part that matters. A fixed lambda has to be tuned to the
    distance between the real and fake distributions, but that distance shrinks
    as the generator learns, so a lambda that was correct at step 0 comes to
    dominate the Wasserstein term later and the critic's gradient vanishes.
    Scaling by W~ makes the constraint fade out on the same schedule.
    """
    eps = torch.rand(real.shape[0], 1, 1, device=real.device)
    x_hat = (eps * real + (1.0 - eps) * fake).requires_grad_(True)
    d_hat = critic(x_hat, y, stage, alpha)
    grad = torch.autograd.grad(
        outputs=d_hat.sum(), inputs=x_hat, create_graph=True
    )[0]
    norm = grad.flatten(1).norm(2, dim=1)
    p1 = torch.clamp(norm - 1.0, min=0.0).pow(2).mean()
    return torch.clamp(w_dist.detach(), min=0.0) * p1


def _spectral_diagnostics(fake, real):
    """
    Compare a fake batch to a real one in the frequency domain.

    Returns (in_band_amplitude_ratio, out_of_band_fraction).

    Total standard deviation is a misleading progress signal here, and was
    actively giving false reassurance: the real trials are band-passed to
    4-40 Hz before training, the generator is not constrained to that band,
    so a generator can match total power exactly while putting most of it
    where the downstream band-pass will delete it. Observed in practice as
    "amplitude x0.92 of real" during training and x0.14 in the saved cache.

    So the band is defined empirically as the frequency bins where the REAL
    batch actually has power, and the ratio is computed only there. That makes
    this number directly comparable to what evaluate_gan.py reports, and it
    works at every stage of the progressive ladder without needing to know the
    stage's effective sample rate - the real data has already been average-
    pooled to that rate, so its own spectrum defines the target support.
    """
    Pf = torch.fft.rfft(fake, dim=2).abs().pow(2).mean(dim=(0, 1))
    Pr = torch.fft.rfft(real, dim=2).abs().pow(2).mean(dim=(0, 1))

    # Define the band as the real spectrum's 99%-energy support: the smallest
    # set of bins that together hold 99% of the real power.
    #
    # A simple threshold like "bins above 1% of the peak" is far too generous
    # here. EEG power falls as ~1/f, so 1% of the 4-8 Hz peak is still above
    # the Butterworth skirts out past 60 Hz, and the mask ends up covering
    # most of the spectrum. That produced "out-of-band 1%" during training for
    # a generator that then lost 78% of its power to the actual 4-40 Hz
    # band-pass - a monitor that reassured instead of warning.
    if Pr.sum() <= 0:
        return float("nan"), float("nan")
    order = torch.argsort(Pr, descending=True)
    keep = int((torch.cumsum(Pr[order], 0) / Pr.sum() < 0.99).sum().item()) + 1
    in_band = torch.zeros_like(Pr, dtype=torch.bool)
    in_band[order[:keep]] = True

    amp_in = torch.sqrt(Pf[in_band].sum() / Pr[in_band].sum().clamp_min(1e-20))
    oob = 1.0 - (Pf[in_band].sum() / Pf.sum().clamp_min(1e-20))
    return amp_in.item(), oob.item()


def stage_steps(steps_per_stage):
    """Generator steps actually run at each stage, after STAGE_STEP_SCALE."""
    return [max(1, int(round(steps_per_stage * s))) for s in STAGE_STEP_SCALE]


def estimate_minutes(steps_per_stage, n_channels=22, batch_size=16,
                     n_critic=5, gp_every=4, device="cpu", probe_steps=2):
    """
    Time a couple of real steps at each resolution and extrapolate.

    Worth the ~20 seconds it costs: the honest answer for this model on CPU is
    hours per generator, and that is something you want to know before you
    start 36 of them, not after.
    """
    device = torch.device(device)
    G = Generator(n_channels=n_channels).to(device)
    D = Critic(n_channels=n_channels).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.0, 0.99))
    optD = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.0, 0.99))

    import time
    total = 0.0
    for stage, (length, n_steps) in enumerate(
            zip(STAGE_LENGTHS, stage_steps(steps_per_stage))):
        x = torch.randn(batch_size, n_channels, length, device=device)
        y = torch.randint(0, 2, (batch_size,), device=device)
        t0 = time.time()
        for i in range(probe_steps):
            for j in range(n_critic):
                z = torch.randn(batch_size, 128, device=device)
                with torch.no_grad():
                    fake = G(z, y, stage, 1.0)
                d_real, d_fake = D(x, y, stage, 1.0), D(fake, y, stage, 1.0)
                w = d_real.mean() - d_fake.mean()
                loss = -w + 1e-3 * d_real.pow(2).mean()
                if j % gp_every == 0:
                    loss = loss + 10.0 * gp_every * _relaxed_gradient_penalty(
                        D, x, fake, y, stage, 1.0, w)
                optD.zero_grad(set_to_none=True)
                loss.backward()
                optD.step()
            z = torch.randn(batch_size, 128, device=device)
            lg = -D(G(z, y, stage, 1.0), y, stage, 1.0).mean()
            optG.zero_grad(set_to_none=True)
            lg.backward()
            optG.step()
        total += (time.time() - t0) / probe_steps * n_steps
    return total / 60.0


class EMA:
    """Exponential moving average of generator weights.

    Samples are always drawn from the averaged weights. A WGAN generator
    oscillates around its optimum rather than settling on it, so the raw
    final iterate is a worse model than the average of recent iterates.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.step = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        self.step += 1
        # Warmup. A fixed decay of 0.999 has a ~1000-step memory, so on any
        # run shorter than several thousand generator steps the average is
        # still dominated by the RANDOM INITIALISATION it started from - and
        # since samples are drawn from the EMA weights, the cached synthetic
        # data is then mostly noise no matter how well the live generator
        # trained. That was measured here: the live generator reached
        # amplitude x0.99 of real while the EMA-sampled cache sat at x0.14.
        # Ramping the decay in makes the average track the live weights
        # closely at first and settle to `decay` once there is enough history.
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)


def train_gan(
    X, y,
    steps_per_stage=1200,
    fade_frac=0.5,
    batch_size=16,
    latent_dim=128,
    lr=1e-3,
    betas=(0.0, 0.99),
    n_critic=5,
    lambda_gp=10.0,
    gp_every=4,
    drift_eps=1e-3,
    up_mode="cubic",
    aug_policy=DEFAULT_POLICY,
    aug_target=0.4,
    val_frac=0.12,
    device="cpu",
    seed=0,
    verbose=True,
    log_every=400,
):
    """
    Train one conditional progressive WGAN-GP on a single subject's trials.

    Parameters
    ----------
    X : (trials, channels, samples) float array, already normalised by
        data.to_gan_space and channels-first. `samples` must be <= 896; it is
        zero-padded to the 896 ladder and cropped back on sampling.
    y : (trials,) labels in {1, 2}
    steps_per_stage : generator steps at the base resolution. Each stage gets
        this scaled by STAGE_STEP_SCALE.
    fade_frac : fraction of each stage (after the first) spent fading the new
        block in.
    gp_every : apply the gradient penalty on every Nth critic step, with
        lambda scaled up by the same factor. This is StyleGAN2's lazy
        regularisation. The penalty needs a double backward through the critic
        and costs roughly as much as the rest of the critic step put together,
        while the constraint it enforces changes slowly - so evaluating it a
        quarter as often at 4x the weight is very nearly free accuracy-wise
        and close to a 2x speedup. Set to 1 for the textbook behaviour.
    aug_target : target for the adaptive augmentation controller, on the r_v
        overfitting statistic described below. Above the target, augmentation
        strength rises; below, it falls.

    The r_v overfitting signal
    --------------------------
    Karras et al.'s better-known heuristic is r_t = E[sign(D(real))], which
    works for a non-saturating GAN because that discriminator's logits are
    calibrated around zero when it cannot tell real from fake. A WGAN critic
    has no such calibration - its output is an unbounded score, and in
    practice it sits entirely on one side of zero, so r_t pins to -1 or +1 and
    the controller never moves. (This was a live bug here: augmentation
    strength decayed to zero on every run and the main small-data safeguard
    was silently switched off.)

    So we use their other, scale-free heuristic instead:

        r_v = (E[D(real_train)] - E[D(real_val)])
              / (E[D(real_train)] - E[D(fake)])

    That is the fraction of the critic's real-vs-fake gap which is explained
    by train-vs-unseen-real rather than by real-vs-fake - i.e. by memorisation.
    r_v near 0 means the critic treats unseen real trials exactly like the
    ones it trained on; r_v near 1 means it separates them as strongly as it
    separates fakes, which is pure memorisation.

    `real_val` is carved out of THIS generator's own training portion (see
    `val_frac`), never from the gate's held-out fold. Tuning augmentation
    against the gate's validation trials would leak them back into the
    synthetic data and undo the whole point of the per-fold generators.

    Returns
    -------
    generator : Generator with EMA weights loaded, in eval mode
    history : dict of per-stage diagnostics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)

    n_trials, n_channels, n_samples = X.shape
    full_len = STAGE_LENGTHS[-1]
    if n_samples > full_len:
        raise ValueError(f"{n_samples} samples exceeds the {full_len} ladder top")

    # Reflect-pad up to 896 so no stage sees an abrupt zero edge that the
    # critic could use as a free real/fake tell.
    pad = full_len - n_samples
    X_pad = np.pad(X, ((0, 0), (0, 0), (0, pad)), mode="reflect") if pad else X

    X_t = torch.as_tensor(X_pad, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(np.asarray(y) - 1, dtype=torch.long, device=device)

    # Internal ADA validation split. Stratified, taken from this generator's
    # own trials, and excluded from the critic's training batches so that the
    # r_v statistic measures something real. Kept small (~12%) because every
    # trial withheld is a trial the generator does not learn from.
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(n_trials, dtype=bool)
    for c in np.unique(y):
        idx = np.where(np.asarray(y) == c)[0]
        n_v = max(2, int(round(len(idx) * val_frac)))
        val_mask[rng.choice(idx, size=min(n_v, len(idx) - 2), replace=False)] = True
    tr_idx = torch.as_tensor(np.where(~val_mask)[0], device=device)
    va_idx = torch.as_tensor(np.where(val_mask)[0], device=device)
    if verbose:
        print(f"    ADA split: {len(tr_idx)} critic-train / {len(va_idx)} "
              f"held out for the r_v overfitting statistic", flush=True)

    G = Generator(n_channels=n_channels, latent_dim=latent_dim,
                  up_mode=up_mode).to(device)
    D = Critic(n_channels=n_channels).to(device)
    # torch >= 2.13 rejects a mixed int/float betas tuple, and (0, 0.99) is the
    # natural way to type Karras' setting, so coerce rather than trip on it.
    betas = (float(betas[0]), float(betas[1]))
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
    ema = EMA(G)

    aug_strength = 0.0
    r_v = 0.0
    history = {"stage": [], "w_dist": [], "aug": [], "r_v": [], "amp": [],
               "amp_inband": [], "oob": []}

    critic_step = 0
    for stage, stage_len in enumerate(STAGE_LENGTHS):
        n_steps = max(1, int(round(steps_per_stage * STAGE_STEP_SCALE[stage])))
        n_fade = 0 if stage == 0 else int(n_steps * fade_frac)
        every = max(1, log_every if log_every <= n_steps else n_steps // 3 or 1)
        # Downsample the real data to this stage's resolution by average
        # pooling, matching the critic's own downsample operator.
        factor = full_len // stage_len
        X_stage = F.avg_pool1d(X_t, factor) if factor > 1 else X_t

        w_run, rv_run, n_run = 0.0, 0.0, 0
        for step in range(n_steps):
            alpha = 1.0 if n_fade == 0 else min(1.0, (step + 1) / n_fade)

            # --- critic ---
            for _ in range(n_critic):
                critic_step += 1
                idx = tr_idx[torch.randint(0, len(tr_idx), (batch_size,),
                                           device=device)]
                real = X_stage[idx]
                yb = y_t[idx]

                z = torch.randn(batch_size, latent_dim, device=device)
                with torch.no_grad():
                    fake = G(z, yb, stage, alpha)

                real_a = diff_augment(real, aug_policy, aug_strength)
                fake_a = diff_augment(fake, aug_policy, aug_strength)

                d_real = D(real_a, yb, stage, alpha)
                d_fake = D(fake_a, yb, stage, alpha)
                w_dist = d_real.mean() - d_fake.mean()

                loss_d = -w_dist + drift_eps * d_real.pow(2).mean()
                if critic_step % gp_every == 0:
                    gp = _relaxed_gradient_penalty(
                        D, real_a, fake_a, yb, stage, alpha, w_dist
                    )
                    loss_d = loss_d + lambda_gp * gp_every * gp

                optD.zero_grad(set_to_none=True)
                loss_d.backward()
                optD.step()

            # --- overfitting statistic r_v, and the augmentation controller ---
            # Measured once per generator step rather than per critic step:
            # it needs a third forward pass and it moves slowly.
            if aug_policy:
                with torch.no_grad():
                    vb = va_idx[torch.randint(0, len(va_idx), (batch_size,),
                                              device=device)]
                    d_val = D(diff_augment(X_stage[vb], aug_policy, aug_strength),
                              y_t[vb], stage, alpha)
                    gap = (d_real.mean() - d_fake.mean()).item()
                    # A non-positive gap means the critic is not separating at
                    # all yet; r_v is undefined there, so hold it steady.
                    if gap > 1e-6:
                        raw = (d_real.mean() - d_val.mean()).item() / gap
                        r_v = 0.9 * r_v + 0.1 * float(np.clip(raw, 0.0, 1.0))

                aug_strength += (0.01 if r_v > aug_target else -0.01)
                aug_strength = float(np.clip(aug_strength, 0.0, 0.85))

            # --- generator ---
            z = torch.randn(batch_size, latent_dim, device=device)
            yb = y_t[torch.randint(0, n_trials, (batch_size,), device=device)]
            fake = G(z, yb, stage, alpha)
            loss_g = -D(diff_augment(fake, aug_policy, aug_strength),
                        yb, stage, alpha).mean()

            optG.zero_grad(set_to_none=True)
            loss_g.backward()
            optG.step()
            ema.update(G)

            w_run += w_dist.item()
            rv_run += r_v
            n_run += 1
            if verbose and (step + 1) % every == 0:
                print(
                    f"    stage {stage} (len {stage_len:4d})  "
                    f"step {step+1:5d}/{n_steps}  "
                    f"W~ {w_run/n_run:8.3f}  r_v {rv_run/n_run:5.2f}  "
                    f"aug {aug_strength:4.2f}",
                    flush=True,
                )
                w_run, rv_run, n_run = 0.0, 0.0, 0

        # --- end-of-stage diagnostics -----------------------------------
        # The fastest early read on whether this run is going anywhere. A
        # doomed run is visible in minutes instead of at the end of the budget.
        # Sampled from the EMA weights, not the live ones. The EMA weights are
        # what sample() ships, and early in training the two differ a lot -
        # monitoring the live generator reported amplitude x0.99 while the
        # EMA-sampled cache was at x0.14. A monitor has to watch the artefact
        # that actually gets used.
        with torch.no_grad():
            zc = torch.randn(min(64, n_trials), latent_dim, device=device)
            yc = y_t[torch.randint(0, n_trials, (zc.shape[0],), device=device)]
            live = {k: v.detach().clone() for k, v in G.state_dict().items()}
            ema.copy_to(G)
            G.eval()
            fake_c = G(zc, yc, stage, 1.0)
            G.load_state_dict(live)
            G.train()
            amp = (fake_c.std() / X_stage.std()).item()
            amp_in, oob = _spectral_diagnostics(fake_c, X_stage)

        history["stage"].append(stage)
        history["w_dist"].append(float(w_dist.item()))
        history["aug"].append(float(aug_strength))
        history["r_v"].append(float(r_v))
        history["amp"].append(float(amp))
        history["amp_inband"].append(float(amp_in))
        history["oob"].append(float(oob))
        if verbose:
            print(f"    stage {stage} done (len {stage_len:4d}): "
                  f"amp x{amp:.2f} total / x{amp_in:.2f} in-band, "
                  f"out-of-band {oob*100:3.0f}%, r_v {r_v:.2f}, "
                  f"aug {aug_strength:.2f}", flush=True)

    ema.copy_to(G)
    G.eval()
    G._n_samples = n_samples
    G._latent_dim = latent_dim
    return G, history


@torch.no_grad()
def sample(generator, n_per_class, n_samples=None, device="cpu",
           batch_size=64, seed=None):
    """
    Draw synthetic trials, balanced across the two classes.

    Returns
    -------
    Z : (2*n_per_class, channels, n_samples) in GAN space (normalised)
    y : (2*n_per_class,) labels in {1, 2}
    """
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device(device)
    generator = generator.to(device).eval()
    n_samples = n_samples or getattr(generator, "_n_samples", STAGE_LENGTHS[-1])
    latent_dim = getattr(generator, "_latent_dim", 128)
    stage = len(STAGE_LENGTHS) - 1

    labels = torch.cat([
        torch.zeros(n_per_class, dtype=torch.long),
        torch.ones(n_per_class, dtype=torch.long),
    ]).to(device)

    out = []
    for i in range(0, labels.shape[0], batch_size):
        yb = labels[i:i + batch_size]
        z = torch.randn(yb.shape[0], latent_dim, device=device)
        out.append(generator(z, yb, stage, 1.0).cpu().numpy())

    Z = np.concatenate(out, axis=0)[:, :, :n_samples]
    return Z.astype(np.float64), (labels.cpu().numpy() + 1)
