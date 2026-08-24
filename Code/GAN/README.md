# GAN Data Augmentation for BCI IV-2a Left/Right Motor Imagery

Design notes for `Code/GAN/`. For how to run it, see
`GAN_execution_commands.txt` at the repository root.

## The problem this has to solve

The existing stacked ensemble reaches **0.72 mean kappa** on Left vs Right. It
is a strong classical pipeline — filter-bank CSP plus three Riemannian models
plus an RBF SVM, across three motor-imagery windows, stacked by a
logistic-regression meta-learner on out-of-fold predictions.

The binding constraint is not the classifier. It is that after artefact
rejection each subject has **113–138 training trials**, roughly 55–69 per
class, in 22 channels. Every model in the ensemble is estimating either a
22×22 spatial covariance per frequency band or a set of CSP filters, from that.

That is the case for generative augmentation, and it is also the reason it is
dangerous. Fitting a generative model of 22×875 EEG on 130 examples is a
textbook setup for a generator that memorises its training set and re-emits it
with jitter. Such a generator would improve every fidelity metric, would pass
a naive TSTR check, would inflate any cross-validated score computed on the
training session, and would contribute nothing — while, in a clinical context,
carrying a real patient's signal inside data labelled "synthetic".

So the engineering problem is two problems: build a generator good enough to
be worth using, and build a decision procedure that can tell whether it is.

## What was taken from the papers

Reviewed from `GAN Research papers/`; the ones that changed the design:

**Hartmann et al. 2018, EEG-GAN** (`1806.01875v1.pdf`) — the reference result
for generating *raw* EEG rather than spectrogram images, and the source of
most of `eeg_wgan.py`:

- Progressive growing (Karras et al. 2017). They start at 24 time samples and
  double six times to 768. We start at 28 and double five times to 896, then
  crop to our 875-sample window. Training the full window from scratch on ~110
  trials does not converge; growing it does.
- **Interpolation upsampling, never transposed convolution.** Their section 2.2
  measures the aliasing directly: nearest-neighbour upsampling injects strong
  high-frequency artefacts, linear and cubic inject far weaker ones. This is
  not cosmetic for us — our filter bank runs to 40 Hz, so generator-borne
  aliasing lands inside the features the classifiers read. We default to cubic
  and expose `--up-mode nearest` only to reproduce their comparison.
- Generator block = upsample + 2 convs of kernel 9; critic block = 2 convs +
  average-pool; LeakyReLU throughout; equalised learning rate, pixel norm,
  minibatch standard deviation; `n_critic=5`, `Adam(1e-3, betas=(0.0, 0.99))`,
  `lambda=10`, drift term `0.001·E[D(x_r)²]`.
- Their improvement to WGAN-GP: a **one-sided** gradient penalty that is
  additionally **scaled by the current critic difference**. Their argument is
  that the correct `lambda` depends on the distance between the real and fake
  distributions, and that distance shrinks as the generator learns — so a fixed
  `lambda` eventually dominates the Wasserstein term and the critic's gradient
  vanishes. In their table, plain WGAN-GP collapsed and this did not.
- Their evaluation conclusion, which shaped `gan_metrics.py`: **no single
  metric is trustworthy.** Their best-FID model produced the *least* realistic
  spectra, and the Inception Score gave no signal about a collapsed model. They
  recommend reading FID, sliced Wasserstein and Euclidean nearest-neighbour
  distance together.

**Suemitsu & Nambu 2023** (`Effects_of_Data_Including_Visual_Presentation...`)
— the methodological warning. Classifiers on IV-2a gain accuracy from the
visual-cue and rest periods rather than from motor imagery, which is how
90 %-plus numbers get published on this dataset. Our windows start 0.5 s after
cue onset, which is the standard MI window, but the 2a cue arrow stays on
screen until 3.25 s. `data.STRICT_MI_WINDOW` is the cue-free alternative
(1.25 s–4.0 s post-cue) if the conservative protocol is wanted; note that
switching to it changes the baseline too, so the 0.72 reference would need
re-running for comparison.

**Bhat & Hortal 2021** (`GANForEEGGeneration.pdf`) — tested augmenting at 2×,
3× and 4×. Combined with the general finding that excessive synthetic data
makes classifiers overfit generator artefacts, this is why the candidate pool
tops out at 2× and why the amount is selected rather than fixed.

**Miyato & Koyama 2018** — projection discriminator. Not in the folder, but
needed: Hartmann trained one GAN per class, which at 55 trials per class is
not viable, so we use a single conditional model that shares all filters
across left and right. Concatenating the label lets a critic ignore it at this
sample size; an inner product with the final feature vector does not.

**Zhao et al. DiffAugment / Karras et al. ADA** — also not in the folder, and
the single most important addition. See below.

## The three design decisions that matter

### One generator per subject, at the widest window

The ensemble looks at three windows: Early (2.5–4.5 s), Full (2.5–6.0 s), Late
(3.5–6.0 s). All three are exact sub-crops of the Full window, so we generate
875 samples once and crop.

Generating each window with its own GAN would let the three views of the
"same" synthetic trial contradict each other — Early saying one thing, Late
saying another about a trial that has no underlying reality to be consistent
with. A meta-learner trained on 15 base models across those windows is exactly
the machinery that would find and exploit that inconsistency.

### Adaptive differentiable augmentation

At ~110 trials the critic memorises the real set within a few hundred steps.
From then on it tells the generator "be one of these 110 signals". The loss
curves look fine while this happens.

`diffaug.py` applies the same differentiable augmentation to real *and* fake
batches before they reach the critic. The critic then has to discriminate
distributions-under-augmentation, which it cannot do by memorising, and
because the ops are differentiable the generator still gets clean gradients.
Critically the augmentation is only seen by the critic, so the generator's
output distribution stays unbiased — unlike augmenting the training set.

Strength is controlled by Karras et al.'s `r_t` heuristic (the mean sign of
the critic's output on real data, a direct read on how much it is
overfitting): above target, strength rises; below, it falls.

The ops are chosen to be label-preserving for motor imagery — time shift,
amplitude and per-channel gain, channel dropout, noise, cutout. Channel
permutation and time reversal are deliberately excluded: both destroy the
C3/C4 lateralisation that is the entire signal being modelled.

### The decision procedure

This is the part that makes the result defensible.

1. **The E session is untouchable.** `train_gan.py` never opens it. The GAN,
   the normalisation statistics and the augmentation choice all come from the
   T session. E is read once, at the end.

2. **The gate is leak-free.** `train_gan.py` trains, per subject, one "full"
   generator plus one generator per gate fold, each fitted without its own
   fold's trials. Fold *k* of the selection CV is scored using synthetic data
   from the generator blind to fold *k*.

   The shortcut — one generator on all of T, then cross-validate the ratio on
   T — leaks: the fakes added to fold *k*'s training portion came from a model
   that had already seen fold *k*'s validation trials. The gate would
   reliably vote for more synthetic data, and the safeguard would be theatre.
   The same folds are reused for the meta-learner's out-of-fold stage, so the
   blind generators apply there too.

3. **"none" is the default and has to be beaten convincingly.** The winner is
   chosen by the one-standard-error rule: a candidate must clear the
   un-augmented baseline by more than one standard error of the fold scores.
   Taking the plain argmax over six candidates on ~45 validation trials would
   itself be overfitting. Several subjects coming out as `none` is the
   safeguard working.

4. **The GAN competes against free alternatives.** Segment-recombination
   (Lotte 2015; Schirrmeister et al. 2017) and Riemannian geodesic
   re-colouring are in the same pool. Riemannian re-colouring is the sharpest
   comparison: it moves a real trial to a genuinely new, geometrically valid
   point on the SPD manifold — a covariance interpolated between two real ones
   — while keeping the real waveform. Since four fifths of the ensemble
   consumes spatial covariance, that populates exactly the region the
   classifiers care about, at millisecond cost and with no memorisation
   failure mode. A GAN that cannot beat it should not be used.

Synthetic trials only ever enter base-learner training sets. The meta-learner
is always fitted on out-of-fold predictions for real trials — a meta-learner
calibrated on synthetic data would be learning how the base models behave on
fakes, which is not the question being asked.

## The audit panel

`evaluate_gan.py` reports fidelity (PSD, mu/beta band-power error,
tangent-space Fréchet distance, sliced Wasserstein) against the training set
the generator was fitted on, and memorisation plus utility (nearest-neighbour
ratio, TSTR/TRTS) per fold against trials the generator never saw. Asking the
second pair against training data would be meaningless — a memorising
generator scores perfectly on both.

The metric to read first is **`nn`**, the nearest-neighbour distance ratio:
mean distance from a fake to its nearest real trial, over the mean distance
between two real trials.

- `≈ 1.0` — fakes sit at the same spacing as reals. Target.
- `< 0.7` — the generator is re-emitting training trials. Failed run.
- `≫ 1` — off-manifold; undertrained.

Nothing in the panel feeds into the final model. The decision is made by
downstream accuracy on held-out real trials; the panel is diagnosis and audit
trail.

## Honest expectations

Published gains from GAN augmentation on motor imagery are largest when the
downstream classifier is a data-hungry deep network on a small training set.
Here the downstream classifier is a well-regularised Riemannian ensemble
already at 0.72 kappa — shrinkage-estimated covariances, elastic-net logistic
regression, a parameter-free MDM. Those are precisely the methods that degrade
most gracefully with few trials, which means there is less headroom for
augmentation to recover.

A realistic outcome is a small gain on a subset of subjects and `none` on the
rest. The pipeline is built so that this outcome is *visible* rather than
hidden: run `main.py --no-gan` and `main.py --all-candidates` and compare. If
the GAN does not clear the classical baseline, that is a legitimate finding at
110 trials per subject, and it is what the gate is designed to surface rather
than paper over.

## Known limitations

- **The gate uses a proxy model.** Scoring the full 15-model stack for every
  candidate × fold would put a single subject into the hours, so the gate uses
  FBCSP+LDA and filter-bank tangent-space+LR soft-voted on the Full window.
  The two families are the most complementary in the ensemble, but the proxy
  can rank candidates slightly differently from the full stack. The
  one-standard-error rule absorbs small mis-rankings by refusing to move off
  the baseline for small differences.
- **Cross-subject transfer is not attempted.** Every generator is
  within-subject. A subject-independent generator fine-tuned per subject (the
  REVG approach, `Residual-Enhanced_VAE-GAN...pdf`) would see 9× the data and
  is the most promising next step, but it changes the evaluation protocol and
  was out of scope here.
- **Band-limited baseline.** This pipeline band-passes everything to 4–40 Hz,
  including the un-augmented baseline, so that the augmented-covariance model
  cannot separate real from synthetic on out-of-band content. That makes the
  `none` row here the band-limited variant of the stacked ensemble rather than
  a bit-identical rerun of `Code/StackedEnsemble/main.py`. All candidates are
  compared against that same baseline, so the internal comparison is
  like-for-like, but quote the two numbers separately.
- **Cue-period caveat** as described under Suemitsu & Nambu above.
