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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffaug import diff_augment, DEFAULT_POLICY

# Resolution ladder. 896 = 28 * 2**5; the canonical window is 875 samples, so
# the last 21 samples are cropped off after generation.
STAGE_LENGTHS = [28, 56, 112, 224, 448, 896]
STAGE_CHANNELS = [128, 128, 96, 80, 64, 48]
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
            return self.to_signal[0](h)

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
        return out


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


class EMA:
    """Exponential moving average of generator weights.

    Samples are always drawn from the averaged weights. A WGAN generator
    oscillates around its optimum rather than settling on it, so the raw
    final iterate is a worse model than the average of recent iterates.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)


def train_gan(
    X, y,
    steps_per_stage=1200,
    fade_frac=0.5,
    batch_size=32,
    latent_dim=128,
    lr=1e-3,
    betas=(0.0, 0.99),
    n_critic=5,
    lambda_gp=10.0,
    drift_eps=1e-3,
    up_mode="cubic",
    aug_policy=DEFAULT_POLICY,
    aug_target=0.6,
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
    steps_per_stage : generator steps at each of the 6 resolutions.
    fade_frac : fraction of each stage (after the first) spent fading the new
        block in.
    aug_target : target for the adaptive augmentation controller. This is
        Karras et al.'s r_t heuristic: the mean sign of the critic's output on
        real data, which measures how confidently the critic separates real
        from fake, i.e. how much it is overfitting. Above the target,
        augmentation strength rises; below, it falls.

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

    G = Generator(n_channels=n_channels, latent_dim=latent_dim,
                  up_mode=up_mode).to(device)
    D = Critic(n_channels=n_channels).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
    ema = EMA(G)

    aug_strength = 0.0
    history = {"stage": [], "w_dist": [], "aug": [], "rt": []}

    def real_batch(bs):
        idx = torch.randint(0, n_trials, (bs,), device=device)
        return X_t[idx], y_t[idx]

    for stage, stage_len in enumerate(STAGE_LENGTHS):
        n_fade = 0 if stage == 0 else int(steps_per_stage * fade_frac)
        # Downsample the real data to this stage's resolution by average
        # pooling, matching the critic's own downsample operator.
        factor = full_len // stage_len
        X_stage = F.avg_pool1d(X_t, factor) if factor > 1 else X_t

        w_run, rt_run = 0.0, 0.0
        for step in range(steps_per_stage):
            alpha = 1.0 if n_fade == 0 else min(1.0, (step + 1) / n_fade)

            # --- critic ---
            for _ in range(n_critic):
                idx = torch.randint(0, n_trials, (batch_size,), device=device)
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

                gp = _relaxed_gradient_penalty(
                    D, real_a, fake_a, yb, stage, alpha, w_dist
                )
                loss_d = -w_dist + lambda_gp * gp + drift_eps * d_real.pow(2).mean()

                optD.zero_grad(set_to_none=True)
                loss_d.backward()
                optD.step()

                # Karras r_t: fraction of real samples the critic scores
                # positive. 0 = cannot tell, 1 = perfectly separated.
                rt = d_real.sign().mean().item()

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

            # --- adaptive augmentation ---
            if aug_policy:
                aug_strength += (0.01 if rt > aug_target else -0.01)
                aug_strength = float(np.clip(aug_strength, 0.0, 0.85))

            w_run += w_dist.item()
            rt_run += rt
            if verbose and (step + 1) % log_every == 0:
                n = log_every
                print(
                    f"    stage {stage} (len {stage_len:4d})  "
                    f"step {step+1:5d}/{steps_per_stage}  "
                    f"W~ {w_run/n:8.3f}  r_t {rt_run/n:5.2f}  "
                    f"aug {aug_strength:4.2f}",
                    flush=True,
                )
                w_run, rt_run = 0.0, 0.0

        history["stage"].append(stage)
        history["w_dist"].append(float(w_dist.item()))
        history["aug"].append(float(aug_strength))
        history["rt"].append(float(rt))

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
