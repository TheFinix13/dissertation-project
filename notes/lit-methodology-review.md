# Chapter 2 literature: how they actually built and trained their models

Compiled 2026-09-07 (post-Nguyen-meeting redo). Sources verified against the
papers and, for TDQN, the authors' published source code. Purpose: check our
implementation choices against what the cited studies actually did, and list
what is worth adopting.

## The comparison

| Study | Model | Algorithm family | Data & frequency | Training-data volume strategy | Training details |
|---|---|---|---|---|---|
| Moody & Saffell 2001 | **Single-layer recurrent net** (position feeds back; tens of parameters) | Actor-only ("direct reinforcement", RRL) | Monthly S&P 500/T-bill 1970–94; intraday USD/GBP FX 1996 | **Tiny model for tiny data**; 25 years of monthly bars is enough for a single layer | Online gradient ascent (RTRL) on differential Sharpe ratio; transaction costs inside the objective |
| Deng et al. 2017 (FDRNN) | Fuzzy representation + deep autoencoder + **recurrent** layer | Actor-only (direct RL, task-aware BPTT) | Chinese index futures, **minute-level** bars | **Higher frequency** = hundreds of thousands of bars | ~1,000 epochs; fuzzy front end to denoise; the deep net learns features from raw price diffs |
| Jiang et al. 2017 (EIIE) | **CNN / RNN / LSTM** variants over 50-period price windows | Deterministic policy gradient on log returns | 12 cryptocurrencies, **30-minute** bars | **Higher frequency** + online stochastic batch learning (geometric sampling of recent windows) | Adam 3e-5; portfolio-vector memory for cost-aware batching; online retraining during test |
| Théate & Ernst 2021 (TDQN) | **MLP: 5 hidden layers × 512**, batch-norm, dropout 0.2 | Value-based (DQN + target net, replay) | 30 stocks/indices, **daily** bars (6y train / 2y test) | **Training "entirely based on generation of artificial trajectories"**: shift, stretch, low-pass filter, noise augmentation of the historical series | Adam 1e-4, L2 1e-6, grad-clip 1, γ=0.4, replay 100k, batch 32, ε exponential decay, Xavier init |
| Guan & Liu 2021 (XRL) | Post-hoc feature attribution on trained portfolio agents | (explains, doesn't train) | — | — | Relevant to our behavioural analysis, not to training |
| FinRL (Liu et al. 2020) | SB3 **MLP defaults** over engineered features | DQN/PPO/A2C/SAC wrappers | Daily stocks | Library leaves it to the user | Standard SB3 training loops |
| Fischer 2018 (survey) | — | Critic-only most common in the field; recurrent nets common when inputs are raw price windows | Daily most common | — | — |

## The three lessons

**1. Nobody trains a deep network on raw daily bars of one asset.** Every deep
study gets its sample volume from somewhere: minute/30-minute frequency (Deng,
Jiang), decades of history with a *single-layer* model (Moody & Saffell), or
generated/augmented data (Théate & Ernst — the closest setup to ours: DQN,
daily bars, discrete actions — trained **entirely on artificial
trajectories**). Our redo (calibrated generator + 64→32 net) is squarely
inside the field's standard responses, and TDQN is the same-domain precedent
to cite for it.

**2. Architecture follows the feature pipeline, not fashion.** LSTM/CNN appear
when the network consumes raw price windows and must extract temporal features
itself (Deng's autoencoder, Jiang's 50-bar windows). When features are
engineered (TDQN's returns/deltas, FinRL, our rolling features), an MLP is the
standard choice. Our 9 engineered rolling features play the role of Deng's
learned front end, so an MLP is aligned with the literature; an LSTM would
add parameters exactly where we can least afford them and duplicate what
mom_k/vol_k/ma_gap already encode. No cited trading study uses GANs for
training data (TimeGAN exists in the wider literature); our transparent
regime generator is the controllable alternative and is what the control
experiment requires.

**3. Capacity discipline is old news.** Moody & Saffell beat Q-learning at
monthly frequency with a single-layer policy — small capacity matched to
small data, in 2001. TDQN gets away with 5×512 *because* augmentation gives
it effectively unlimited data. Nguyen's parameters-vs-episodes argument is
the same principle; the literature has obeyed it all along.

## What we already do that matches

- Simulated/augmented training data with real held-out transfer test (TDQN).
- MLP over engineered features (TDQN, FinRL).
- Small network matched to data volume (Moody & Saffell's principle).
- Transaction costs inside the reward (all of them).
- Multi-seed reporting (beyond most of the cited studies).

## Worth adopting if time allows (validation-gated, in priority order)

1. **TDQN-style augmentation of the real episodes** (shift/noise/filter) as a
   middle rung between 60 real episodes and pure simulation. Separates "was it
   volume or balance?" and directly mirrors the closest prior work.
2. **PPO entropy coefficient > 0** (currently SB3 default 0) — the likely fix
   for the corner-policy collapse; select on the simulated validation split.
3. **TDQN training hygiene**: dropout/L2, gradient clipping, Xavier init.
   Cheap, standard, defensible. Their γ=0.4 is also notable: short effective
   horizon for daily trading; ours is 0.99.
4. **Differential Sharpe reward** (Moody & Saffell) — already in ch6 future
   work as the risk-efficiency objective; the citation is in the bib.

## What not to do

- LSTM/CNN policy networks: wrong side of the capacity budget, duplicates the
  engineered features, and unsupported by our data volume.
- GAN-generated training data: no precedent in the cited trading studies,
  opaque, and would surrender the regime control that motivates simulation.
