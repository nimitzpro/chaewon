# Does Roughness Matter? Signature-MMD Diagnostics for Stochastic Volatility Model Comparison

## MSc Thesis Proposal — Outline

---

## 1. Motivation

The "rough volatility" debate is one of the most active controversies in quantitative finance:

- **Gatheral, Jaisson & Rosenbaum (2018)** found that the Hurst exponent of realised volatility is approximately $H \approx 0.1$, far below the $H = 0.5$ of standard diffusions — launching the rough volatility programme and models like rough Bergomi.
- **Cont & Das (2022)** challenged this: they show that realised volatility of *any* continuous semimartingale appears rough ($H < 0.5$), even when the underlying instantaneous volatility is a standard diffusion. The observed roughness may be an artefact of the estimator, not the latent process.

Both sides traditionally use **static calibration** (fitting single-day implied volatility surfaces) as their primary evidence. Recent literature highlights that not only is volatility path-dependent (Guyon & Lekeufack, 2023), but the implied volatility surface itself exhibits strong path-dependence (Andrès, Boumezoued, & Jourdain, 2026). However, because models with very different underlying dynamics can still be calibrated to produce nearly identical static IV snapshots, static calibration alone cannot definitively resolve the debate. This thesis proposes a **path-distributional** approach: after calibrating models to the same IV surface, we ask whether their *simulated path dynamics* are distinguishable.

## 2. Core Idea

Use **signature-kernel Maximum Mean Discrepancy (sig-MMD)** — a kernel-based two-sample test on path space (Chevyrev & Oberhauser 2022; Salvi et al. 2021) — as a **post-calibration diagnostic** applied to a hierarchy of stochastic volatility models spanning the Markovian → path-dependent → rough spectrum.

The key question: *when models are calibrated to reproduce the same implied volatility surface, are their path-space distributions distinguishable?*

**Why sig-MMD?**
- It is a proper metric on distributions of stochastic processes (not just terminal distributions)
- It captures the full path dynamics, including features like volatility clustering, leverage effects, and the Zumbach effect — features that, while driving the path-dependent evolution of IV surfaces in real markets, cannot be distinguished by single-day static IV fits
- The signature kernel is computed efficiently via a Goursat PDE (Salvi et al. 2021), with GPU implementations available (`sigkerax`)
- It is model-agnostic: the same diagnostic applies to all models regardless of their mathematical structure

## 3. Model Hierarchy

Six models spanning the full Markovian → path-dependent → rough spectrum:

| # | Model | Type | Pricing |
|---|---|---|---|
| 0 | Heston | Classical Markovian (baseline) | Fourier (Lewis) |
| 1 | 1F Quintic OU | Modern Markovian | Fourier-Laplace |
| 2 | 2F Quintic OU | Modern Markovian (multi-factor) | Fourier-Laplace |
| 3 | PD-Bergomi | Path-dependent (lifted Markov) | Fourier-Laplace |
| 4 | 4F-PDV (Guyon) | Path-dependent (functional) | MC |
| 5 | Rough Bergomi | Genuinely rough | MC |

**Why these models?** Heston is the baseline. Quintic OU (Abi Jaber & Li 2024) represents the state-of-the-art Markovian approach; the 1F model uses a single OU factor with a quintic polynomial vol function, while the 2F adds a second factor for joint SPX/VIX calibration. PD-Bergomi bridges Markovian and rough via its shift parameter $\varepsilon$ (as $\varepsilon \to 0$ it approaches rBergomi). The 4F-PDV of Guyon provides path-dependence *without* roughness. Rough Bergomi is the canonical rough model. Together, they let us test whether it is roughness, path-dependence, or something else that drives dynamic behaviour.

## 4. Methodology

1. **Calibrate** each model to SPX implied volatility data using its best-available method (Fourier/AD for Fourier-tractable models; MC + IV RMSE for the rest)
2. **Simulate** paths from each calibrated model
3. **Compute** the pairwise sig-MMD distance matrix across all models, with permutation tests for statistical significance
4. **Compare** simulated model paths against observed historical market paths (daily SPX + VIX), ranking models by closeness to real market dynamics

## 5. Experiments

1. **Synthetic validation:** Verify sig-MMD detects known differences between models in controlled conditions using canonical (published) parameters
2. **Single-date SPX calibration:** Calibrate all models to one SPX date; confirm comparable IV fits
3. **Cross-model distance matrix (core):** Compute the full $6 \times 6$ pairwise sig-MMD matrix after calibration — the central thesis figure. Apply MDS to visualise the model topology in path space
4. **Dynamic calibration:** Weekly recalibration over 2017–2022; track how pairwise model distances evolve over time
5. **Regime analysis:** Stratify by market regime (calm, crisis, rate-hiking) — in which regimes does roughness become detectable?
6. **Real market paths (core, novel):** Compare each model's simulated paths against observed historical SPX dynamics via sig-MMD. This is the most distinctive contribution — no prior study has compared stochastic volatility models against real market paths in path-distributional space

## 6. Expected Contributions

- A **path-distributional perspective** on the "rough or not" debate, independent of $H$-estimation artefacts (addressing the Cont & Das critique directly)
- The first **model-vs-market** comparison using sig-MMD on real historical paths
- An empirical **topology of stochastic volatility models** in path space — revealing whether the Markovian / path-dependent / rough distinction is sharp or continuous
- Evidence on whether path-dependence (Guyon's mechanism) or roughness (fractional kernels) better explains observed market dynamics

## 7. Key References

**The rough volatility debate:**
- Gatheral, Jaisson & Rosenbaum (2018). "Volatility is rough." *Quantitative Finance* 18(6). arXiv:1410.3394
- Cont & Das (2022). "Rough volatility: fact or artefact?" arXiv:2203.13820

**Signature-kernel methods:**
- Chevyrev & Oberhauser (2022). "Signature moments to characterize laws of stochastic processes." *JMLR* 23(176). arXiv:1810.10971
- Salvi, Cass, Foster, Lyons & Yang (2021). "The Signature Kernel is the Solution of a Goursat PDE." *SIAM J. Math. Data Sci.* arXiv:2006.14794
- Issa, Horvath, Lemercier & Salvi (NeurIPS 2023). "Non-adversarial training of Neural SDEs with signature kernel scores." arXiv:2305.16274

**Models:**
- Abi Jaber & Li (2024). "Volatility models in practice: Rough, Path-dependent or Markovian?" arXiv:2401.03345
- Andrès, Boumezoued, & Jourdain (2026). "The implied volatility surface (also) is path-dependent." *Quantitative Finance* 1-31
- Bayer, Friz & Gatheral (2016). "Pricing under rough volatility." *Quantitative Finance* 16(6)
- Guyon & Lekeufack (2023). "Volatility is (mostly) path-dependent." *Quantitative Finance* 23(9)
- Gazzani & Guyon (2024). "Pricing and calibration in the 4-factor PDV model." arXiv:2406.02319

**Statistical testing:**
- Alden, Horvath & Issa (2025). "Signature Maximum Mean Discrepancy Two-Sample Statistical Tests." arXiv:2506.01718

## 8. Implementation

- **Language/framework:** Python (JAX for GPU-accelerated simulation and sig-MMD computation via `sigkerax`)
- **Hardware:** RTX 3090 (24GB VRAM)
- **Data:** CBOE SPX options data; daily SPX + VIX for historical path construction
- **Timeline:** ~16 weeks (model wrappers → calibration → experiments → write-up)
