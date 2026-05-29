# Does Roughness Matter? Signature-MMD Diagnostics for Stochastic Volatility Model Comparison
## Full Implementation Roadmap

---

## Project Overview

**Thesis claim:** We use signature-kernel MMD (Chevyrev & Oberhauser 2022; Issa et al. NeurIPS 2023) as a **diagnostic and model-comparison metric** applied *after* calibration to a hierarchy of 8 stochastic volatility models spanning the full spectrum from classical Markovian through path-dependent to genuinely rough, rough+ (stochastic vol-of-vol), and signature-based (non-parametric). Each model is calibrated via its best-available method (Fourier/AD for Fourier-tractable models, MC+AD for path-dependent and rough models, linear regression on signature features for the Sig-Vol model, IV RMSE as universal baseline). The sig-MMD metric is then used to answer a question that no static calibration loss can: *when models are calibrated to reproduce the same implied volatility surface, are their path-space distributions distinguishable?* At which point on the Markovian → path-dependent → rough spectrum do models become dynamically distinguishable? In which market regimes does roughness — or path-dependence — become detectable? **Crucially, we also compare simulated model paths against observed historical market paths** (daily SPX returns + realised/implied volatility), directly testing which model best reproduces the dynamic behaviour of the S&P 500 — a comparison no prior study has attempted using path-distributional methods. **An additional methodological question:** including a signature-based model (Alòs et al. 2025) alongside the sig-MMD diagnostic allows us to test whether the diagnostic is biased toward models that share its mathematical basis — the C2ST diagnostic (which uses no signatures) provides the independence check.

**Motivation — the "rough or not" debate:** The thesis situates itself squarely within the ongoing controversy over whether volatility is genuinely rough or merely appears so:
- Gatheral, Jaisson & Rosenbaum (2018) found H ≈ 0.1 from high-frequency realised vol, launching rough volatility.
- **Cont & Das (2022)** challenged this: they show realised volatility *always* appears rough (H < 0.5) even when instantaneous vol is standard-diffusive — the roughness may be an artefact of the realized volatility estimator, not the latent process.
- **Fukasawa, Takabatake & Westphal (2019)** developed a quasi-likelihood estimator for H from noisy data; they confirm roughness but warn about estimation bias.
- **Rosenbaum & Zhang (2022)** show a universal LSTM and rough vol models agree on volatility forecasting, arguing universality from both parametric and nonparametric angles.
- **Abi Jaber & Li (2024)** demonstrate that Markovian Quintic OU models can match SPX IV surfaces as well as rough Bergomi, questioning whether roughness is operationally necessary.
- **Guyon & Lekeufack (2023) / Gazzani & Guyon (2024)** show that a path-dependent but Markovian 4-factor PDV model fits SPX and VIX jointly — an entirely different resolution of the same puzzle.

All these studies compare models via static metrics (IV RMSE, forecasting RMSE, characteristic function fit). **No study compares calibrated stochastic volatility models in path space** — the dynamic, distributional level where rough and Markovian models should differ most. Moreover, no study compares *simulated model paths* against *observed historical market paths* using path-distributional methods. This thesis fills both gaps: we compare models against each other and against reality.

**Prior art boundary:** Issa et al. (2023) use sig-MMD to fit *nonparametric neural SDEs* to generic time-series data; we use sig-MMD as a post-calibration diagnostic on *parametric* models. No prior work applies sig-MMD to compare named stochastic volatility models structurally. The closest work is:
- Teng & Li (arXiv:2512.00448, 2025): Wasserstein-1 distributional calibration of rBergomi — but W₁ operates on terminal distributions only, not path distributions.
- Zhu et al. (arXiv:2007.02113, 2020): Markovian approximation of rBergomi for pricing, not distributional comparison.
- Alòs et al. (arXiv:2507.23392, 2025): signatures as a *model representation* with standard IV RMSE loss — different paradigm.
- Alden, Horvath & Issa (arXiv:2506.01718, 2025): sig-MMD two-sample test theory — the statistical backbone we use for hypothesis testing.

**Sig-MMD literature map:**

| Paper | Role in this thesis |
|---|---|
| Chevyrev & Oberhauser (JMLR 2022, arXiv:1810.10971) | **Mathematical foundation** — proves sig-moments characterise laws of stochastic processes; derives the sig-MMD metric on path space; provides the non-parametric two-sample test. Issa et al. build directly on this. |
| Salvi, Lemercier et al. (NeurIPS 2021, arXiv:2109.03582) | **Why sig-MMD > standard MMD** — shows that standard KME/MMD is blind to the filtration (the flow of information); higher-order KME captures it. Justifies using sig-MMD over a naive path-feature MMD. |
| Issa, Horvath, Lemercier & Salvi (NeurIPS 2023, arXiv:2305.16274) | **Direct prior art** — sig-MMD for neural SDE training; proves strict properness + Goursat PDE backprop. Foundation for this thesis. |
| Alden, Horvath & Issa (2025, arXiv:2506.01718) | **Concurrent** — formalises sig-MMD as a statistical two-sample test; discusses Type II error (false null acceptance) and mitigation. Directly relevant to experimental design: use their permutation test to validate that sig-MMD detects model differences. |
| Lemercier, Lyons & Salvi (2024, arXiv:2404.02926) | **Practical improvement** — log-PDE method for sig kernels on rough paths; reduces first-order Goursat PDE errors for rough Bergomi paths specifically. |
| Redhead et al. (2026, arXiv:2602.10182) | **Application** — sig-MMD + censored sig-MMD for probabilistic forecasting evaluation. Shows sig-MMD is gaining traction as an evaluation metric. |

**Models under study (7):**

| # | Model | Type | Kernel / mechanism | Vol fn | Pricing | Params |
|---|---|---|---|---|---|---|
| 0 | Heston (baseline) | Classical Markovian | — (CIR process) | $\sqrt{V_t}$ | Fourier (Lewis) | $\kappa, \theta, \xi, \rho, V_0$ |
| 1 | 1F Quintic OU | Modern Markovian | $e^{-\lambda t}$ | $p(X)$ deg 5 | Fourier-Laplace | $\lambda, \rho, \alpha_0, \alpha_1, \alpha_3, \xi_0(\cdot)$ |
| 2 | 2F Quintic OU | Modern Markovian (multi-factor) | $e^{-\lambda_x t}, e^{-\lambda_y t}$ | $p(Z)$ deg 5 | Fourier-Laplace | $\lambda_x, \lambda_y, \omega, \rho, \alpha_k, \xi_0(\cdot)$ |
| 3 | PD-Bergomi | Path-dependent (lifted Markov) | $(t+\varepsilon)^{H-1/2}$ (shifted frac.) | $e^X$ | Fourier-Laplace | $\nu, H, \varepsilon, \rho, \xi_0(\cdot)$ |
| 4 | 4F-PDV (Guyon) | Path-dependent (functional) | — (EWMA of returns + RV) | $f(\bar{R}^{(1)}, \bar{R}^{(2)}, \bar{V}^{(1)}, \bar{V}^{(2)})$ | MC only | $a_0, a_R^{(i)}, a_V^{(i)}, h_R^{(i)}, h_V^{(i)}, \rho$ |
| 5 | Rough Bergomi | Genuinely rough | $t^{H-1/2}$ (fractional) | $e^X$ | MC only | $\eta, H, \rho, \xi_0(\cdot)$ |
| 6 | Grey Bergomi | Rough + stochastic vol-of-vol | Grey BM kernel | $e^X$ | MC only | $\eta, H, \gamma, \rho, \xi_0(\cdot)$ |
| 7 | Sig-Vol (Alòs et al.) | Signature-based (non-parametric) | Truncated path signature | $\sigma_t = \langle \ell, S^{\leq m}(\mathbf{x})_{0,t} \rangle$ | Linear regression + MC | $m$ (truncation depth), $a_I$ (sig coefficients) |

> **Notation:** $\nu$ (PD-Bergomi vol-of-vol) and $\eta$ (rBergomi / Grey Bergomi vol-of-vol) play the same role but follow the respective source conventions. Heston $\theta$ is the long-run variance level; the 2F Quintic OU mixing weight is denoted $\omega$ to avoid clash. Quintic OU params: $\alpha_5 = 1$ is fixed (normalisation); free coefficients are $\alpha_0, \alpha_1, \alpha_3$. PD-Bergomi: $\varepsilon > 0$ controls the shift — as $\varepsilon \to 0$ the model approaches rough Bergomi; as $\varepsilon \to \infty$ it becomes effectively Markovian. PDV: $\bar{R}^{(i)}, \bar{V}^{(i)}$ are EWMAs of past log-returns and realized variance with half-lives $h_R^{(i)}, h_V^{(i)}$. Grey Bergomi: $\gamma$ controls the deviation from standard fBm — when $\gamma = 0$, Grey Bergomi reduces to rough Bergomi; $\gamma > 0$ introduces stochastic vol-of-vol via generalised grey Brownian motion.

> **Design rationale — why this hierarchy?** The 8 models span a deliberate spectrum: Heston is the baseline. Quintic OU models represent the state-of-the-art Markovian approach (Abi Jaber & Li 2024). PD-Bergomi bridges Markovian and rough via the shift parameter $\varepsilon$. The 4F-PDV of Guyon provides an entirely different resolution — path-dependence without roughness, using functional dependence on recent history. Rough Bergomi is the "rough" anchor. Grey Bergomi (Jacquier, Oliveri Orioles & Zuric 2025) extends rBergomi with stochastic vol-of-vol via generalised grey Brownian motion, achieving joint SPX/VIX calibration that plain rBergomi cannot. Sig-Vol (Alòs et al. 2025) sits *off* the Markovian → rough axis entirely: volatility is a linear functional of the truncated path signature — a non-parametric, data-driven approach that does not commit to any roughness value. Including it tests whether sig-MMD is biased toward models sharing its mathematical basis (signatures), with C2ST providing the independence check. The sig-MMD distance matrix will reveal the topology of this spectrum.

**Calibration methods (4) — role clarification:**

| # | Method | Role in thesis | Works for all models? |
|---|---|---|---|
| A | IV RMSE | **Primary calibration** (universal baseline) | Yes (requires pricing map per model) |
| B | Fourier/AD | **Primary calibration** for Fourier-tractable models (gold standard) | No (Heston, 1F/2F Quintic OU, PD-Bergomi only) |
| C | Sig-MMD via Goursat PDE | **Post-calibration diagnostic** — measures path-space model distance; also used as secondary calibration loss to show feasibility | **Yes** (model-agnostic) |
| D | Neural surrogate | Stretch goal — speed comparison | Yes (but requires offline training per model) |

> **Key shift from earlier framing:** Sig-MMD is not "the" calibration engine. Each model is calibrated by the fastest available method. Sig-MMD is then applied to the calibrated model paths to answer questions that IV RMSE cannot: are the models dynamically distinguishable after static calibration?

### 📚 Study Material — Project Overview & Background

This section covers the prerequisite knowledge for the entire thesis. Work through in order.

**1. Stochastic volatility models — foundations**
- **Bergomi (2015), Ch. 1–3** (*Stochastic Volatility Modeling*, CRC Press). The canonical reference for stochastic vol from the practitioner perspective. Ch. 1 covers why Black-Scholes is insufficient, Ch. 2 introduces local and stochastic vol, Ch. 3 covers the Heston model and its Fourier pricing. You need this background before anything else.
- **Gatheral (2006)** (*The Volatility Surface: A Practitioner's Approach*, Wiley). Shorter and more intuitive than Bergomi. Read Ch. 1–5 for: what the volatility surface is, why it exists, how models generate different surface shapes, and SVI parameterisation. Good for building intuition about what calibration is trying to match.
- **Shreve (2004)** (*Stochastic Calculus for Finance II*, Springer), Ch. 4–6, if your measure theory / Itô calculus is rusty. You need: Itô's formula, Girsanov's theorem (risk-neutral measure), the Feynman-Kac theorem (connection between PDEs and expectations).

**2. The "rough or not" debate — read in this order**
- **Gatheral, Jaisson & Rosenbaum (2018)** (arXiv:1410.3394). *Read first.* The founding empirical paper. Focus on §2 (the log-variogram estimator and why it gives $H \approx 0.1$), §3 (the scaling argument), and Figure 1 (the straight line in log-log coordinates). Understand: they estimate $H$ from the *realised* volatility, not the latent process.
- **Cont & Das (2022)** (arXiv:2203.13820). *Read second.* The key counterargument. Focus on Theorem 1 (realised vol of *any* continuous semimartingale appears rough) and the simulation study in §4. The central insight: the roughness Gatheral et al. measure could be an artefact of the realised vol estimator.
- **Abi Jaber & Li (2024)** (arXiv:2401.03345). *Read third.* The practical resolution for pricing: Markovian Quintic OU models fit the SPX IV surface as well as rough Bergomi. Focus on Table 1 (RMSE comparison) and the concluding discussion. This motivates the thesis question: if Markovian models fit equally well *statically*, do they differ *dynamically*?
- **Guyon & Lekeufack (2023)** (*Quantitative Finance* 23(9)). *Read fourth.* The "third way" — path-dependence without roughness. Focus on the R² analysis showing that past returns and past realized variance explain >90% of next-day variance. This is the empirical basis for the 4F-PDV model.
- **Fukasawa, Takabatake & Westphal (2019)** (arXiv:1905.04852), **Rosenbaum & Zhang (2022)** (arXiv:2206.14114), **Zarhali et al. (2025)** (arXiv:2505.02678) — read selectively for additional nuance. These are important for the literature review but not prerequisites for implementation.

**3. Signatures & sig-MMD — the mathematical toolkit**
- **Lyons, Caruana & Lévy (2007)** (*Differential Equations Driven by Rough Paths*, Springer LNM 1908), Ch. 1–2 only. This is the original rough paths monograph. You need: what a path signature is (iterated integrals), Chen's identity, and the universal nonlinearity theorem. Don't read the full book — just enough to understand why signatures characterise paths.
- **Chevyrev & Oberhauser (2022)** (arXiv:1810.10971). *The core theory paper.* Read Theorem 1 (expected signature characterises the law of a stochastic process), §3 (the signature MMD), and §4 (the two-sample test). This is the theoretical backbone of the thesis.
- **Kidger & Lyons (2021)** "Signatory: differentiable computations of the signature and logsignature transforms, on both CPU and GPU." arXiv:2001.00706. Not directly used (we use sigkerax instead), but the clearest *tutorial-style* introduction to what signatures compute. Read §2 for the best concise explanation of truncated signatures, their algebraic structure, and computational complexity.
- **Salvi et al. (2021)** (arXiv:2006.14794). Read for the Goursat PDE derivation — how $k(\mathbf{x}, \mathbf{y})$ satisfies a PDE instead of needing explicit signature computation.
- **Issa et al. (NeurIPS 2023)** (arXiv:2305.16274). The direct prior art. Focus on §2 (sig-kernel scores), §3 (strict properness proof — why sig-MMD is a valid loss), and §5.2 (the rough volatility experiment — they calibrate a neural SDE to rough Bergomi paths using sig-MMD).

**4. MMD — the statistical framework**
- **Gretton et al. (2012)** "A kernel two-sample test." *JMLR* 13. arXiv:0805.2368. **The foundational MMD paper.** Read §2 (MMD definition), §3 (the unbiased estimator), §4 (the permutation test), §5 (consistency and power). You must understand: what MMD measures, why it is zero iff distributions are equal (for characteristic kernels), and how the permutation test works. This is prerequisite for everything in Phase 3.
- **Briol et al. (2019)** (arXiv:1906.05944). The extension to generative models. Read §2 (MMD as a scoring rule) and §3 (consistency of MMD-minimising estimators). Relevant for understanding why sig-MMD calibration (Step 6) produces consistent parameter estimates.
- **Alden, Horvath & Issa (2025)** (arXiv:2506.01718). Read for the specific sig-MMD permutation test used in Experiment 3. Focus on §3 (the test procedure) and §4 (Type II error analysis — when will the test fail to distinguish models?).

**5. Fourier pricing — how calibration actually works**
- **Lewis (2001)** (*Option Valuation under Stochastic Volatility*, Finance Press), Ch. 1–4. The classic. Or equivalently, **Carr & Madan (1999)** "Option valuation using the fast Fourier transform," *J. Computational Finance* 2(4). Either source covers: characteristic functions, the Fourier inversion formula for call prices, and numerical integration via FFT. You need this to understand how Heston calibration works.
- **Abi Jaber, Li & Lin (2024)** (arXiv:2405.02170). The Fourier-Laplace extension for polynomial OU models. Read §2 (the Riccati framework) and §3 (numerical implementation). This covers pricing for Quintic OU and PD-Bergomi.

**Suggested reading order (minimum viable):**
1. Gatheral (2006) Ch. 1–5 or Bergomi (2015) Ch. 1–3 → stochastic vol basics
2. Gatheral, Jaisson & Rosenbaum (2018) → the rough vol hypothesis
3. Cont & Das (2022) → the counter-argument
4. Gretton et al. (2012) → MMD fundamentals
5. Chevyrev & Oberhauser (2022) §1–4 → sig-MMD theory
6. Issa et al. (2023) §2–5 → sig-MMD in practice
7. Abi Jaber & Li (2024) → the model comparison paper that most directly motivates this thesis

---

## Phase 0: Infrastructure (Week 1–2)

### 0.1 Environment Setup

**Primary compute machine:** i9-11900k + RTX 3090 (24 GB VRAM), accessed remotely via VSCode + Tailscale. All heavy simulation and calibration runs there.

**Local testing machine:** MacBook (Apple Silicon). CPU-only JAX works out of the box; use this for prototyping, unit tests, and small-scale smoke tests before pushing to the remote machine. The same code runs on both — only the `pip install` line differs.

---

#### Remote machine (Linux + RTX 3090)

```bash
# Verify CUDA version first
nvidia-smi  # should show CUDA 12.x for RTX 3090 with recent drivers

# Core stack — CUDA 12 build
pip install "jax[cuda12]"  # installs jaxlib-cuda12 automatically
pip install diffrax sigkerax optax equinox
pip install matplotlib scipy pandas quadax yfinance
```

> **CUDA 11 fallback:** If `nvidia-smi` shows CUDA 11.x, use `pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html` instead. RTX 3090 shipped with CUDA 11 support but drivers from 2023+ typically default to CUDA 12.

> **Remote dev workflow:** With VSCode + Tailscale, connect via the Remote-SSH extension. The VSCode Python interpreter should point to the remote virtualenv. The JAX XLA compilation cache persists between runs on the remote machine — set `XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"` if JAX can't find CUDA automatically. Run `JAX_PLATFORM_NAME=gpu python -c "import jax; print(jax.devices())"` to confirm GPU is detected.

---

#### Local Mac (Apple Silicon — for prototyping and testing)

```bash
# CPU-only JAX — works on all Apple Silicon Macs, no extra config
pip install jax  # CPU backend, no extra suffix needed
pip install diffrax sigkerax optax equinox
pip install matplotlib scipy pandas quadax yfinance

# Optional: Metal GPU acceleration (experimental — JAX 0.4.14+)
# pip install jax-metal
# Note: jax-metal is still maturing; some ops fall back to CPU silently.
# Verify: python -c "import jax; print(jax.devices())"
# CPU-only gives: [CpuDevice(id=0)]
# With jax-metal: [METAL(id=0)]
```

> **M-series performance:** Apple Silicon's unified memory means `jax-metal` can be faster than CPU for moderate batch sizes (N ≤ 512 paths). For N ≥ 1024 or long paths (≥ 100 steps), the RTX 3090 will be ~10–20× faster. Use the Mac for correctness checks and the remote for production runs.

> **Tip:** Keep a `conftest.py` or env var check that sets small defaults (N=32 paths, 20 steps) when `JAX_PLATFORM_NAME=cpu` is detected, so test runs finish in seconds on the Mac without changing experiment code.

### 0.2 Project Structure
```
thesis/
├── models/
│   ├── __init__.py
│   ├── heston.py          # Thin wrapper around existing implementation
│   ├── quintic_1f.py      # 1F Quintic OU: single OU + quintic polynomial (~20 lines); OR 2F repo with θ=1
│   ├── quintic_2f.py      # Wrapper around shaunlinz02/two_factor_quintic_ou (2F only repo)
│   ├── pd_bergomi.py      # Wrapper: sum-of-exponentials approx via existing rBergomi code
│   ├── pdv_4f.py          # ~30-line diffrax.Euler for the 6D SDE (no existing JAX impl)
│   ├── rough_bergomi.py   # Wrapper around rbergomi package (Pakkanen)
│   ├── grey_bergomi.py    # Extension of rBergomi with grey BM (Jacquier et al. 2025)
│   └── sig_vol.py         # Signature-based vol model (Alòs et al. 2025); linear functional of truncated sig
├── diagnostics/           # THE CORE — where the thesis contribution lives
│   ├── __init__.py
│   ├── sig_kernel.py      # sigkerax wrapper for Goursat PDE
│   ├── sig_mmd.py         # MMD computation, pairwise distance matrix, permutation test
│   ├── c2st.py            # Classifier two-sample test (LSTM/1D-CNN discriminator)
│   ├── stylized_facts.py  # Scorecard: vol clustering, leverage, kurtosis, Zumbach, vol-of-vol
│   ├── p_variation.py     # Empirical p-variation of log-vol paths per model
│   └── model_map.py       # MDS/t-SNE embedding + visualisation
├── calibration/
│   ├── __init__.py
│   └── iv_rmse.py         # IV RMSE wrapper (call existing calibrators, store params)
├── data/
│   ├── synthetic.py       # Generate ground-truth rBergomi paths for Experiment 1
│   ├── market.py          # Load CBOE/OptionMetrics SPX data
│   └── forward_variance.py # Bootstrap ξ₀(t) from market
├── experiments/
│   ├── exp1_synthetic.py       # Synthetic validation
│   ├── exp2_single_date.py     # Single-date SPX calibration
│   ├── exp3_cross_model.py     # Cross-model distance matrix (CORE)
│   ├── exp4_dynamic.py         # Dynamic calibration + evolving distances (CORE)
│   ├── exp5_regimes.py         # Regime-dependent distinguishability (CORE)
│   └── exp6_real_paths.py      # Sig-MMD vs real market paths (CORE — PRIORITY)
├── notebooks/
│   ├── 01_model_validation.ipynb
│   ├── 02_sigkerax_experiments.ipynb  # Hyperparameter sensitivity: path count, kernel, lead-lag
│   ├── 03_static_results.ipynb
│   └── 04_dynamic_results.ipynb
├── tests/
│   ├── test_wrappers.py   # Smoke-test each model wrapper: shapes, constraints, reproducibility
│   └── test_sig_kernel.py # Validate sig-MMD: positivity, symmetry, known distances
└── thesis/
    └── main.tex
```

> **Key architectural decision:** All model wrappers expose a single interface:
> ```python
> def simulate(params: dict, key: jax.Array, n_paths: int, n_steps: int, T: float) -> jnp.ndarray:
>     """Returns shape (n_paths, n_steps+1, d) where d = state dimension."""
> ```
> The `diagnostics/` layer only ever sees this interface — it is completely model-agnostic. This separation means the thesis contribution (sig-MMD pipeline) is independent of which simulator is used.

### 0.3 Data Acquisition

#### The core problem: what are the "target paths" in sig-MMD?

Sig-MMD compares simulated model paths against a *target distribution of paths*. The model side is easy — you simulate. The market side requires a deliberate design choice between two options:

---

**Option A: Historical time-series paths (dynamic view)**

Target paths are observed $(t, \log S_t, \hat{\sigma}_t)$ windows from historical daily data.

| Variable | Source | Ticker | Notes |
|---|---|---|---|
| $S_t$ (SPX daily closes) | Yahoo Finance | `^GSPC` | Free, ~70 years |
| $\hat{\sigma}_t$ (coarse proxy) | Yahoo Finance | `^VIX` | VIX as $\sigma_t$ proxy (see caveat below) |
| $\hat{\sigma}_t$ (term structure) | CBOE | VIX9D, VIX, VIX3M, VIX6M | Richer vol proxy |
| Risk-free rate | Yahoo Finance | `^IRX` | 3m T-bill; flat approximation acceptable |

Path construction: use rolling 30-day windows of $(t, \log S_t, \text{VIX}_t)$ — gives ~200 target paths per year.

What this calibrates to: the **dynamic distributional behaviour** of $(S_t, \sigma_t)$ over time. This is information IV RMSE completely discards, making it the genuinely novel use of sig-MMD.

> **VIX ≠ instantaneous vol caveat:** VIX is a 30-day risk-neutral *expected* integrated variance, not the instantaneous spot vol $\sigma_t$ in the models. This mismatch must be acknowledged in the thesis. Mitigation: (1) treat VIX as a noisy proxy and discuss the bias; (2) for Quintic OU models, use the explicit VIX² formula (Prop. 2.4 of 2212.08297) to simulate model-implied VIX paths for a like-for-like comparison; (3) consider using realised vol (e.g., 5-min RV from Oxford-Man Institute) as an alternative $\hat{\sigma}_t$ proxy.

---

**Option B: IV surface-as-path (static cross-sectional view)**

Target paths are the SPX implied vol surface organised as a curve over strikes per maturity.

| Variable | Source | Notes |
|---|---|---|
| SPX option mid IV | CBOE delayed quotes | cboe.com/delayed_quotes/spx/ — free, single date |
| SPX option IV (time series) | OptionMetrics via WRDS | Check university library access |
| Forward prices per maturity | Derived from put-call parity | |
| Risk-free rate per maturity | Yahoo `^IRX` or SOFR | Flat rate fine for thesis |

Path construction: for each maturity $T_j$, lift the smile as a 1D curve:
$$\mathbf{y}^{(j)} = \big((k_1, \sigma_{\text{IV}}(k_1, T_j)),\; (k_2, \sigma_{\text{IV}}(k_2, T_j)),\; \ldots\big) \in \mathbb{R}^2$$
Stack maturities to get the full surface as a path in $\mathbb{R}^3$ parametrised by $(\tau, k)$.

What this calibrates to: the **static shape of the IV surface** on one date — directly comparable to IV RMSE.

---

**Which option to use per experiment:**

| Experiment | Option | Reason |
|---|---|---|
| Synthetic recovery (Exp. 1) | Neither — simulate rBergomi as target | No market data needed |
| Static SPX calibration (Exp. 2) | **Option B** | Direct comparison to IV RMSE baseline |
| Dynamic calibration (Exp. 4) | **Option A** | Naturally time-series; daily IV surfaces harder to obtain |
| Cross-model sig distance (Exp. 3) | Neither — compare calibrated model paths directly | |

---

**Practical recommendation:** start with Option A (Yahoo only — immediately available) to build the sig-MMD pipeline. Add Option B (CBOE) for the static comparison experiment.

**Minimum data pull:**
```python
import yfinance as yf

spx  = yf.download("^GSPC", start="2011-01-01", end="2025-12-31")["Close"].squeeze()
vix  = yf.download("^VIX",  start="2011-01-01", end="2025-12-31")["Close"].squeeze()
rate = yf.download("^IRX",  start="2011-01-01", end="2025-12-31")["Close"].squeeze()
```

This is sufficient for Option A and for bootstrapping the forward variance curve $\xi_0(t)$.

---

### 0.4 Validation Infrastructure
Each wrapper needs a **smoke test only** — shape, dtype, no NaNs, parameter constraints respected. Do not replicate paper benchmarks from scratch; instead, confirm that wrapped existing code reproduces the paper's published numbers (spot-check one number per model).

- **Heston**: price one call with known params → compare to Lewis formula
- **Quintic OU**: one IV slice → compare to Abi Jaber & Li (2024) reported RMSE order of magnitude
- **PD-Bergomi**: at large $\varepsilon$, paths should look Markovian; at $\varepsilon \to 0$, should match rBergomi variance
- **4F-PDV**: vol level and autocorrelation plausible; compare to Gazzani & Guyon Table 1
- **Rough Bergomi**: log-variance ACF should show power-law decay consistent with H ≈ 0.1
- **Grey Bergomi**: at $\gamma = 0$, paths should match rBergomi exactly; at $\gamma > 0$, vol-of-vol should increase (check variance of variance paths across ensembles)

### 📚 Study Material — Infrastructure & Data

**1. JAX ecosystem**
- **JAX documentation** (jax.readthedocs.io): Read the "JAX 101" tutorial series, especially: "Thinking in JAX" (functional programming model), "JIT compilation" (tracing vs. execution), "Automatic differentiation" (forward + reverse mode), and "Pseudo-random numbers" (why JAX uses explicit keys). You will use `jax.grad`, `jax.jit`, `jax.vmap` constantly.
- **Kidger (2022)** "On Neural Differential Equations." PhD thesis, University of Oxford (arXiv:2202.02435). Ch. 2 and Appendix A are the best introduction to `diffrax` — the ODE/SDE solver library used for simulation. Read the "Getting started" section of the diffrax docs (docs.kidger.site/diffrax/) for practical usage: `diffrax.Euler`, `diffrax.Heun`, `diffrax.MultiTerm`, and `diffrax.ControlTerm`.
- **`sigkerax` documentation** (github.com/crispitagorico/sigkerax): Read the README and examples. The key class is `SigKernel` — understand `static_kernel_kind`, `dyadic_order` (controls PDE grid resolution), and how to compute the Gram matrix.

**2. Forward variance curve bootstrapping**
- **Bergomi (2015), §2.5** ("The forward variance curve"). Explains what $\xi_0(t)$ is, why it's an input not a parameter, and how to bootstrap it from variance swaps or from the SVI parameterisation of the IV surface. This is essential for Experiment 2.
- **Gatheral (2006), Ch. 3** ("The SVI parameterisation"). SVI is the most common way to interpolate/extrapolate the IV surface before extracting forward variances. Understand: the 5 SVI parameters, the no-arbitrage constraints (Roper 2010), and how to convert an SVI fit to a forward variance curve.
- **Carr & Wu (2009)** "Variance Risk Premiums." *Rev. Financial Studies* 22(3). For understanding the relationship between variance swaps, VIX, and the forward variance curve. Not strictly necessary but clarifies the VIX ≠ instantaneous vol issue.

**3. Market data & option conventions**
- **CBOE VIX White Paper** (cboe.com/micro/vix/vixwhite.pdf). Read to understand exactly what VIX measures: $\text{VIX}^2 = \frac{2}{\Delta} \sum_i \frac{\Delta K_i}{K_i^2} e^{rT} Q(K_i)$ — a discretisation of the variance swap rate. This is important for understanding why VIX is not the instantaneous vol.
- **Hull (2022)** (*Options, Futures, and Other Derivatives*, 11th ed.), Ch. 19–20. Standard reference for option pricing conventions, put-call parity, and how market data is structured. Skim if you already know this.

---

## Phase 1: Model Wrappers (Week 2–4)

> **Strategy:** Do not reimplement from scratch. The thesis contribution is the sig-MMD diagnostic pipeline, not novel SDE simulation. Wrap existing, tested implementations behind the common `simulate()` interface. The one exception is 4F-PDV, which has no existing JAX implementation — but the SDE is a simple 6D Euler scheme (~30 lines).

### 1.1 Heston — Week 2

**Existing code options (pick one):**
- Simple NumPy Euler-Maruyama (trivial, ~20 lines)
- Any QuantLib Python binding
- `stochvol` or similar

**Wrapper task:** call whichever simulator, convert output to `jnp.array` of shape `(n_paths, n_steps+1, 2)` for $(\log S_t, V_t)$.

**Calibration:** Heston has a well-known closed-form Fourier pricer. Use an existing implementation (QuantLib, or the `py_vollib` / `mibian` ecosystem) to get calibrated params. Store them as a dict. The sig-MMD layer never needs to call the pricer.

**Deliverable:** `models/heston.py` — wrapper + `simulate()`

### 1.2 Path-Dependent Bergomi — Week 2

**Strategy:** Approximate the shifted fractional kernel as a sum of $n=5$–$8$ exponentials. This reduces PD-Bergomi to a multi-factor Bergomi model, which is trivially simulatable as $n$ correlated OU processes.

**Existing code:** Use the `rbergomi` package or any rough Bergomi reference. The sum-of-exponentials weights $(c_i, \lambda_i)$ can be precomputed once for a given $(H, \varepsilon)$ using the Gaussian quadrature of Abi Jaber & El Euch (1908.09999) — this is a ~10-line numerical integration.

**Calibration:** Use any existing calibration code for rough/Bergomi-type models, or simply take the calibrated parameters from Abi Jaber & Li (2024) as starting points and fine-tune.

**Deliverable:** `models/pd_bergomi.py`

### 1.3 4-Factor PDV (Guyon) — Week 2–3

**No suitable existing JAX implementation** — write from scratch, but it is genuinely simple:

```python
def pdv_drift_diffusion(t, state, params):
    log_s, r1, r2, v1, v2 = state
    s = jnp.exp(log_s)
    sigma_sq = (params.a0 + params.aR1*r1 + params.aR2*r2
                          + params.aV1*v1 + params.aV2*v2)**2
    sigma = jnp.sqrt(sigma_sq)
    drift = jnp.array([-0.5*sigma_sq,
                       -r1/params.hR1, -r2/params.hR2,
                       (sigma_sq - v1)/params.hV1,
                       (sigma_sq - v2)/params.hV2])
    diffusion = jnp.array([sigma, 1/params.hR1, 1/params.hR2, 0., 0.])
    return drift, diffusion  # pass to diffrax.Euler
```

**Calibration:** MC + IV RMSE. Start from the parameters published in Gazzani & Guyon (2024, Table 1) — use those directly for Experiment 3 if calibration time is short.

**Deliverable:** `models/pdv_4f.py` — the one model written from scratch (~50 lines)

### 1.4 1-Factor Quintic OU — Week 3

**Existing code:** `github.com/shaunlinz02/two_factor_quintic_ou` implements the **2F model only**. However, the 1F model is exactly the 2F model with $\theta = 1$ (which sets $Z_t = X_t$, making the $Y_t$ factor irrelevant). Use the same repo and fix $\theta = 1$; the $\lambda_y$ parameter becomes unused.

Alternatively, implement the 1F directly from scratch — it is a single OU integral:
```python
# X_t = integral_0^t e^{-lambda(t-s)} dW_s  (exact Gaussian simulation)
# sigma_t = p(X_t) * sqrt(xi_0(t))
# p(x) = alpha_0 + alpha_1*x + alpha_3*x^3 + x^5  (alpha_5=1 fixed)
```
This is ~20 lines in JAX using the known OU covariance structure, and avoids depending on the 2F code path for the simpler model.

**Calibration:** Use calibrated parameters from Abi Jaber et al. (2024) Table 1 directly as starting points. Run short fine-tuning with IV RMSE if needed.

**Deliverable:** `models/quintic_1f.py`

### 1.5 2-Factor Quintic OU — Week 3

**Existing code:** `github.com/shaunlinz02/two_factor_quintic_ou` — this repo implements the 2F model. Clone and use directly.

**Wrapper task:** call the reference simulation, convert to `jnp.array`. If the quintic_1f wrapper is already built as the $\theta=1$ special case of this code, the 2F wrapper is minimal additional work.

**Calibration:** Use calibrated parameters from Abi Jaber & Li (2025) directly as starting points.

**Deliverable:** `models/quintic_2f.py`

### 1.6 Rough Bergomi — Week 3–4

**Existing code:** `rbergomi` package (Pakkanen, pip-installable). Implements the hybrid scheme of Bennedsen, Lunde & Pakkanen (2017).

**Wrapper task:** call `rBergomi.rBergomi(n, N, T, a)`, convert to JAX array.

**Calibration:** Published parameters from Bayer, Friz & Gatheral (2016) / Abi Jaber & Li (2024) as starting points.

**Deliverable:** `models/rough_bergomi.py`

### 1.7 Grey Bergomi — Week 4

**Strategy:** Grey Bergomi (Jacquier, Oliveri Orioles & Zuric, arXiv:2505.08623, 2025) replaces standard fBm with generalised grey Brownian motion (gBm) in the rough Bergomi framework. The key difference: gBm introduces stochastic vol-of-vol controlled by an additional parameter $\gamma$. When $\gamma = 0$, the model reduces to rough Bergomi.

**Existing code:** No existing JAX implementation. However, the simulation is a straightforward extension of rBergomi — replace the fBm kernel with the grey BM kernel (a Mittag-Leffler-type generalisation). The `rbergomi` package can be adapted: the main change is in the covariance structure of the Gaussian process driving log-variance.

**Calibration:** MC + IV RMSE. Start from rBergomi calibrated params ($H, \eta, \rho$) and add $\gamma \approx 0$ as initial guess. Joint SPX/VIX calibration is a key selling point of this model.

**Key question for thesis:** Does the additional stochastic vol-of-vol structure ($\gamma > 0$) produce sig-MMD-detectable path dynamics beyond what plain roughness ($H < 0.5$) provides?

**Deliverable:** `models/grey_bergomi.py`

### 1.8 Sig-Vol (Alòs et al.) — Week 4

**Strategy:** The Sig-Vol model (Alòs, Burés, de Santiago & Vives, arXiv:2507.23392, 2025) defines instantaneous volatility as a linear functional of the truncated path signature:
$$\sigma_t = \sum_{|I| \leq m} a_I \cdot S^I(\mathbf{x})_{0,t}$$
where $S^{\leq m}(\mathbf{x})_{0,t}$ is the truncated signature of the (augmented) price path up to time $t$, and the multi-index coefficients $a_I$ are learned from data. This is a fundamentally different paradigm from all other models in the hierarchy: it is non-parametric and does not commit to a roughness exponent $H$ or a specific SDE form.

**Existing code:** No existing JAX implementation. However, the implementation is lightweight:
1. Compute truncated path signatures using `signatory` or `iisignature` (CPU) or a JAX reimplementation of the signature transform (~50 lines for truncation depth $m \leq 4$)
2. Calibration is linear regression: given simulated price paths and their market-implied volatilities, regress sig features against IV. No MC optimisation loop required.
3. Simulation: generate price paths (e.g., from a base process like GBM or rBergomi), compute sig features, apply the learned linear map to obtain $\sigma_t$.

**Calibration:** Linear regression of truncated signature features against market IV — orders of magnitude faster than MC-based calibration. This is a key advantage: the model is calibrated in seconds, not hours.

**Key question for thesis:** Does sig-MMD "unfairly" favour the Sig-Vol model because both the diagnostic and the model use path signatures as their mathematical foundation? The C2ST diagnostic (which uses no signatures — just an LSTM classifier) provides the independence check: if C2ST agrees with sig-MMD's ranking, the circularity is benign.

**Deliverable:** `models/sig_vol.py`

### 📚 Study Material — Models

**1. Heston model**
- **Heston (1993)** "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." *Rev. Financial Studies* 6(2):327–343. The original paper. Short and readable — defines the model and derives the characteristic function. Focus on: the CIR variance process, the Feller condition, and the closed-form $\phi(u)$.
- **Alfonsi (2015)** *Affine Diffusions and Related Processes: Simulation, Theory and Applications*, Springer, Ch. 3–4. The definitive reference on simulating CIR processes correctly. Key issue: the square-root process can go negative under naive Euler discretisation. Alfonsi covers exact simulation (Broadie & Kaya 2006), reflected/absorbed Euler, and the QE scheme (Andersen 2008). Read §3.2 for the practical schemes.
- **Andersen (2008)** "Efficient simulation of the Heston stochastic volatility model." Working paper. Describes the QE (quadratic-exponential) scheme — the most widely used Heston simulation scheme in practice. Faster and more accurate than Euler for the CIR process.

**2. Quintic OU models**
- **Abi Jaber, Illand & Li (2024)** (arXiv:2212.08297). *The 1F Quintic OU paper.* Read in full — it's well-written. Focus on: §2.1 (the polynomial volatility construction — why quintic?), §2.3 (the VIX formula — Proposition 2.4 is critical), §3 (Fourier-Laplace pricing), and §4 (calibration results). The key idea: a polynomial function of a Gaussian OU process gives tractable moments and Fourier transforms while capturing the vol smile better than affine models.
- **Abi Jaber & Li (2025)** (arXiv:2503.14158). *The 2F extension.* Read §2 (why two factors — the answer is smile dynamics and the skew-stickiness ratio) and §3 (the extended VIX formula). The 2F model achieves joint SPX-VIX calibration that 1F cannot.
- **Abi Jaber, Li & Lin (2024)** (arXiv:2405.02170). *The Fourier-Laplace pricing engine.* Read §2 (the general Riccati framework for polynomial OU volatility) and §3 (numerical solution of the Riccati ODE). This is the pricing method used for both Quintic OU models and PD-Bergomi. Key insight: the polynomial vol structure means the moment-generating function satisfies a *polynomial* Riccati ODE, solvable by standard ODE methods.

**3. Path-Dependent Bergomi**
- **Abi Jaber & El Euch (2019)** (arXiv:1908.09999). *The theoretical foundation.* Read §2 (the Volterra process and its Markovian lift) and §3 (the sum-of-exponentials approximation). Key result: any completely monotone kernel $K(t)$ (which the shifted fractional kernel $(t+\varepsilon)^{H-1/2}$ is for $\varepsilon > 0$) can be written as $K(t) = \int_0^\infty e^{-\lambda t} \mu(d\lambda)$ for some positive measure $\mu$. Discretising this integral gives the sum-of-exponentials approximation: $K(t) \approx \sum_{i=1}^n c_i e^{-\lambda_i t}$. Each exponential component is an independent OU process.
- **Abi Jaber & Li (2024)** (arXiv:2401.03345), §2.1. Defines PD-Bergomi in the context of model comparison. Read for the exact dynamics and the role of $\varepsilon$.
- **Bergomi (2015), Ch. 7–8**. The multi-factor Bergomi framework that PD-Bergomi generalises. Read for: the variance curve dynamics, the log-normal vol structure, and why multi-factor models capture the term structure of vol-of-vol.
- **Carmona, Coulon & Schwarz (2013)** "Approximating Volterra processes." Chapter in *Prokhorov and Contemporary Probability Theory*, Springer. Alternative discussion of sum-of-exponentials approximation for Volterra kernels.

**4. 4-Factor PDV (Guyon)**
- **Guyon & Lekeufack (2023)** (*Quantitative Finance* 23(9)). *Read first.* This is the empirical paper that motivates the model. Focus on: §2 (the R² regression — past returns and realised vol explain >90% of next-day variance), §3 (the EWMA construction), and §4 (the model formulation). The key conceptual point: this model says volatility is *mostly* a deterministic function of recent market history — no latent state needed.
- **Gazzani & Guyon (2024)** (arXiv:2406.02319). *The pricing/calibration paper.* Read §2 (the exact 4-factor dynamics and their SDE representation), §3 (MC pricing), and §4 (calibration to SPX and VIX). Table 1 has the calibrated parameters you can use directly. Focus on understanding why the squaring $\sigma_t^2 = (\ldots)^2$ ensures positivity.
- **Zumbach (2010)** "Volatility conditional on price trends." *Quantitative Finance* 10(4):431–442. The original observation that past returns predict future vol — the empirical regularity that PDV formalises. Read for intuition about the "leverage effect on steroids" that the $\bar{R}$ factors capture.

**5. Rough Bergomi**
- **Bayer, Friz & Gatheral (2016)** (*Quantitative Finance* 16(6)). *The definition paper.* Read in full — it's short. Focus on: §2 (the model dynamics and the fractional kernel $t^{H-1/2}$), §3 (simulation via Cholesky), and §4 (the calibration results). Understand: why $H \approx 0.1$ produces steep short-dated skews that classical models cannot match.
- **Bennedsen, Lunde & Pakkanen (2017)** (*Finance & Stochastics* 21). *The simulation paper.* Read §3 (the hybrid scheme: split the kernel into a near-field power-law and a far-field Riemann-sum component) and §4 (convergence rates). This is the algorithm inside the `rbergomi` package.
- **McCrickerd & Pakkanen (2018)** (arXiv:1708.02563). Read §3 (variance reduction) — antithetic variates on $W$ and conditional MC. Practical necessity for rBergomi simulation (without variance reduction, MC standard errors are ~5× larger).
- **Gatheral, Jaisson & Rosenbaum (2018)** (arXiv:1410.3394), §4. The connection between $H \approx 0.1$ and the power-law behaviour of the ATM skew $\psi(T) \sim T^{H-1/2}$ for small $T$. This is *why* rough models generate steep short-dated skews.

**6. Fractional Brownian motion — prerequisites**
- **Nualart (2006)** *The Malliavin Calculus and Related Topics*, Springer, §1.3 ("Fractional Brownian motion"). Gives the definition, covariance structure, and representation as a Volterra integral of standard BM. Read for the mathematical definition of $\hat{W}_t^H = \int_0^t (t-s)^{H-1/2} dW_s$.
- **Mandelbrot & Van Ness (1968)** "Fractional Brownian Motions, Fractional Noises and Applications." *SIAM Review* 10(4):422–437. The original paper. Read for historical context and intuition about self-similar processes.

**7. Grey Bergomi — extending roughness**
- **Jacquier, Oliveri Orioles & Zuric (2025)** "Rough Bergomi turns grey." arXiv:2505.08623. *The defining paper.* Read §2 (generalised grey Brownian motion — definition, covariance structure, and how it reduces to fBm when $\gamma = 0$), §3 (the Grey Bergomi model dynamics and simulation), and §4 (joint SPX/VIX calibration results). Key insight: grey BM introduces stochastic vol-of-vol *within* the rough framework, which plain rBergomi lacks. This explains why rBergomi struggles with joint SPX/VIX calibration but Grey Bergomi succeeds.
- **Mura & Pagnini (2008)** "Characterizations and simulations of a class of stochastic processes to model anomalous diffusion." *J. Physics A* 41(28):285003. The mathematical foundation for grey Brownian motion. Read if you need the technical details of gBm's covariance structure and M-Wright function.

**8. Sig-Vol — signature-based volatility**
- **Alòs, Burés, de Santiago & Vives (2025)** (arXiv:2507.23392). *The defining paper.* Read §2 (the truncated signature construction — why signatures capture path history), §3 (the linear functional model $\sigma_t = \langle \ell, S^{\leq m} \rangle$ and calibration via linear regression on signature features), and §4 (comparison against Heston and rBergomi using standard IV RMSE). Key insight: the model is non-parametric in the sense that it does not assume a specific SDE for vol — the signature basis functions learn the vol dynamics from data.
- **Cuchiero, Gazzani, Möller & Svaluto-Ferro (2024)** (arXiv:2301.13235). *The richer signature SDE model.* Read §2 (signature SDEs — both drift and diffusion are linear functionals of the signature) and §4 (joint SPX/VIX calibration). Heavier than Alòs et al. but shows the full generality of the signature approach.
- **Lyons (1998)** "Differential equations driven by rough paths." *Rev. Mat. Iberoam.* 14(2):215–310. The foundational reference for path signatures. Read §2 for the definition of the iterated-integrals signature and its key properties (Chen's identity, uniqueness up to tree-like equivalence). Not essential for implementation but deepens understanding of *why* signatures characterise paths.

---

## Phase 2: Calibration Setup (Week 4–6)

> **Strategy:** Calibration is infrastructure, not contribution. The goal is to arrive at a set of calibrated parameter vectors for each model on each date. Use the fastest path to those numbers.

### 2.1 Use published parameters as starting points

For the **single-date experiment (Exp 2)**, the most efficient approach is to start from published calibrated parameters and do a short fine-tune:
- **Quintic OU**: Abi Jaber & Li (2024) publish calibrated params for Oct 23, 2017 SPX — use directly
- **Rough Bergomi**: Bayer, Friz & Gatheral (2016) Table 1 or Abi Jaber & Li (2024)
- **4F-PDV**: Gazzani & Guyon (2024) Table 1 — use directly or fine-tune
- **Heston**: trivial to calibrate fresh (< 1 min with any IV RMSE optimizer)

### 2.2 IV RMSE calibration wrapper — Week 4–5

For models that need re-calibration (weekly dates for Exp 4), implement a thin IV RMSE wrapper:

```python
def calibrate_iv_rmse(model_simulate_fn, market_iv_surface, init_params,
                      n_paths=2048, n_steps=100):
    """Generic IV RMSE calibration via MC pricing.
    Works for any model exposing the simulate() interface."""
    # MC-price the option grid, invert to IV, compute RMSE, gradient-step
    ...
```

For Fourier-tractable models (Heston, Quintic OU, PD-Bergomi), plug in an existing Fourier pricer instead of MC. Do **not** write the Fourier pricer from scratch — use:
- **Heston**: `py_vollib_vectorized`, `QuantLib`, or a 15-line implementation of Lewis (2001)
- **Quintic OU**: use the reference code from `shaunlinz02/two_factor_quintic_ou` if it includes pricing

**Deliverable:** `calibration/iv_rmse.py` — generic wrapper, ~100 lines

### 2.3 Neural Surrogate (drop first if time is short)

Not needed for the core experiments. If implemented, use for 4F-PDV only (the one model where MC calibration is genuinely slow and no Fourier shortcut exists).

**Deliverable:** `calibration/neural_surrogate.py` — stretch goal only

### 📚 Study Material — Calibration

**1. What calibration is**
- **Cont & Tankov (2004)** *Financial Modelling with Jump Processes*, Chapman & Hall, Ch. 13 ("Calibration of parametric models"). The clearest textbook treatment of calibration as an inverse problem. Read §13.1 (the calibration problem), §13.2 (regularisation — why the problem is ill-posed), and §13.3 (practical algorithms). Key idea: calibration is an inverse problem; many different parameter vectors can produce similar IV surfaces (non-identifiability), which is *precisely* why post-calibration diagnostics are needed.
- **Bergomi (2015), Ch. 2.6–2.8**. Practical calibration from the quant desk perspective: choice of objective function (IV RMSE vs. price RMSE vs. Vega-weighted), optimisation methods (Levenberg-Marquardt, differential evolution), and common pitfalls (local minima, overfitting short maturities).

**2. Fourier pricing — the engine behind Heston/Quintic OU calibration**
- **Lewis (2001)** (*Option Valuation under Stochastic Volatility*, Finance Press), Ch. 1–4. The foundational reference. Read Ch. 2 (the generalised Fourier transform for option pricing) and Ch. 3 (application to Heston). The formula: $C(K,T) = S_0 - \frac{\sqrt{S_0 K}}{\pi} \int_0^\infty \text{Re}\big[\frac{\phi(u - i/2; \theta)}{u^2 + 1/4} e^{-iu \log K}\big] du$. Lewis's formulation is numerically superior to Carr-Madan for single strikes.
- **Carr & Madan (1999)** "Option valuation using the fast Fourier transform." *J. Computational Finance* 2(4):61–73. The FFT-based approach — prices a whole grid of strikes simultaneously. Read for the FFT implementation; compare to Lewis for single-strike evaluation.
- **Lord & Kahl (2007)** "Optimal Fourier inversion in semi-analytical option pricing." *J. Computational Finance* 10(4). Read for the practical issue of choosing the integration contour (the "damping factor" $\alpha$) to ensure numerical stability. Important when implementing Fourier pricing.

**3. MC-based calibration — for 4F-PDV and rBergomi**
- **Glasserman (2003)** *Monte Carlo Methods in Financial Engineering*, Springer, Ch. 7 ("Variance reduction techniques"). Read §7.1–7.3 for antithetic variates, control variates, and importance sampling. These are essential for making MC calibration practical — without them, the gradient signal from MC pricing is too noisy.
- **Bayer, Horvath et al. (2019)** (arXiv:1908.08806). The seminal deep calibration paper: train a NN to map parameters → IV surface, then invert via gradient descent. Read §2 (the two-step approach) and §3 (application to rough Bergomi). Relevant as context even though you're using a simpler MC+AD approach.

**4. Automatic differentiation through simulators**
- **Baydin et al. (2018)** "Automatic differentiation in machine learning: a survey." *JMLR* 18(153):1–43. arXiv:1502.05767. Read §2 (forward vs. reverse mode AD) and §4 (AD for scientific computing). You need to understand: why `jax.grad` can differentiate through an entire SDE simulation, and the memory/compute tradeoffs of forward vs. reverse mode.
- The JAX docs on "Custom derivative rules" (jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html) — relevant if you need to handle non-differentiable operations (e.g., the BS inversion step) inside the calibration pipeline.

---

## Phase 3: Calibration & Diagnostic Methods (Week 6–9)

### 3.1 Primary: IV RMSE Calibration — Week 6

**Loss function:**
$$\mathcal{L}_{\text{RMSE}}(\theta) = \sqrt{\frac{1}{N}\sum_{i=1}^{N} \big(\sigma_{\text{IV}}^{\text{model}}(K_i, T_i; \theta) - \sigma_{\text{IV}}^{\text{market}}(K_i, T_i)\big)^2}$$

**Optimisation:**
- For Fourier-tractable models: `optax.adam` with `jax.grad` through the Fourier pricer
- For MC-priced models: use finite-difference gradients or pathwise AD through `diffrax`

**Deliverable:** `calibration/iv_rmse.py`

### 3.2 Primary: Fourier/AD Calibration — Week 6–7

For Markovian models only. The pipeline:
$$\theta \xrightarrow{\text{char. fn}} \phi(u; \theta) \xrightarrow{\text{Lewis inversion}} C(K,T;\theta) \xrightarrow{\text{BS invert}} \sigma_{\text{IV}}(K,T;\theta) \xrightarrow{\text{MSE}} \mathcal{L}$$

AD via `jax.grad` through the entire chain. This is the "gold standard" for Markovian models — fast, exact gradients, no simulation noise. This is the **primary calibration method** for Heston, 1F/2F Quintic OU, and PD-Bergomi (all Fourier-tractable). For 4F-PDV and rough Bergomi (MC-only), use IV RMSE with MC+AD (§3.1).

**Deliverable:** `calibration/fourier_ad.py`

### 3.3 Signature-MMD Diagnostic (THE NOVEL CONTRIBUTION) — Week 7–9

This is the core of the thesis. **Unlike the earlier framing**, sig-MMD is used primarily as a **post-calibration diagnostic metric**, not as the calibration objective. The pipeline:

#### Step 1: Calibrate each model via its best method
Use Fourier/AD (§3.2) for Fourier-tractable models (Heston, 1F/2F Quintic OU, PD-Bergomi). Use IV RMSE with MC+AD (§3.1) for 4F-PDV and rough Bergomi. At this point all models reproduce the market IV surface to comparable accuracy.

#### Step 2: Simulate calibrated-model paths
For each calibrated model, simulate $N$ paths of the augmented process:
$$\mathbf{x}^{(i)} = \big(t, \log S_t, \sigma_t\big)_{t \in [0,T]}, \quad i = 1, \ldots, N$$

#### Step 3: Compute signature kernel via Goursat PDE
Using `sigkerax`, compute the signature kernel matrix:
$$k(\mathbf{x}, \mathbf{y}) = \langle S(\mathbf{x}), S(\mathbf{y}) \rangle$$

where $S(\mathbf{x})$ is the signature of path $\mathbf{x}$ and $k$ satisfies the Goursat PDE:
$$\frac{\partial^2 k}{\partial s \partial t}(s,t) = \langle \dot{\mathbf{x}}_s, \dot{\mathbf{y}}_t \rangle \, k(s,t)$$
with boundary conditions $k(0, \cdot) = k(\cdot, 0) = 1$.

`sigkerax` solves this PDE directly — no truncation of the signature needed.

> **For rough Bergomi paths specifically:** the standard Goursat PDE discretisation incurs first-order errors when paths have low regularity (Hölder exponent $< 1$, which rBergomi paths satisfy for $H < 0.5$). Lemercier, Lyons & Salvi (arXiv:2404.02926) propose a log-PDE method with higher-order accuracy for such paths. Consider using it if sig-MMD shows poor discrimination for rBergomi paths.

> **Scalability alternatives & FDM hybrid diagnostic:** Tamayo-Rios, Schell & Alaifari (arXiv:2502.20392, NeurIPS 2025) propose local Neumann-series expansions of the Goursat PDE that offer better accuracy for rough paths, reduced memory requirements, and scalability to very long time series on a single GPU — a potential drop-in replacement if `sigkerax` proves too slow. Separately, Zhang et al. (arXiv:2410.03973, 2024) propose FDM, an O(D) scoring rule (vs sig-kernel's O(D²)) that exploits the Markov property of SDEs. **FDM is not a replacement for sig-MMD** — it cannot handle non-Markovian models (rBergomi, Grey Bergomi, and arguably PD-Bergomi and 4F-PDV). Instead, use FDM as a **complementary diagnostic for the Markovian subset** (Heston, 1F/2F Quintic OU): compute both sig-MMD and FDM distances for these models and verify they agree. Where they agree, FDM validates sig-MMD; where they disagree, the discrepancy reveals information about non-Markovian path structure that FDM misses. This comparison is a natural sub-experiment within Experiment 3.

#### Step 4: Pairwise model comparison via MMD
For any two calibrated models $\mathcal{M}_a, \mathcal{M}_b$:
$$\text{MMD}^2(\mathcal{M}_a, \mathcal{M}_b) = \frac{1}{N^2}\sum_{i,j} k(\mathbf{x}_a^{(i)}, \mathbf{x}_a^{(j)}) - \frac{2}{N^2}\sum_{i,j} k(\mathbf{x}_a^{(i)}, \mathbf{x}_b^{(j)}) + \frac{1}{N^2}\sum_{i,j} k(\mathbf{x}_b^{(i)}, \mathbf{x}_b^{(j)})$$

This gives a $7 \times 7$ **path-space distance matrix** between calibrated models (plus market paths when available — see Experiment 6).

#### Step 5: Two-sample hypothesis test
Use the permutation-based sig-MMD two-sample test of Alden, Horvath & Issa (arXiv:2506.01718, 2025) to formally test:
- **H₀:** Model A and Model B produce indistinguishable path distributions.
- **H₁:** The models are dynamically distinguishable.

This converts the distance metric into formal statistical statements: e.g., "PD-Bergomi and rBergomi are indistinguishable at the 5% level after calibration to the same IV surface" (or not).

#### Step 6: Optional — sig-MMD as calibration loss
As a **secondary experiment**, also run sig-MMD calibration directly:
$$\theta \xrightarrow{\text{diffrax}} \text{paths} \xrightarrow{\text{sigkerax}} \text{kernel matrix} \xrightarrow{} \text{MMD}^2 \xrightarrow{\text{jax.grad}} \nabla_\theta \text{MMD}^2$$

This demonstrates feasibility of sig-MMD calibration and answers: *does calibrating to path-space (sig-MMD) vs. static surface (IV RMSE) produce different parameter estimates?* Use `optax.adam` with gradient clipping.

#### Practical considerations
- **Path count:** $N = 256$–$1024$ paths per model (memory scales as $O(N^2)$ for kernel matrix)
- **Path length:** 50–200 time steps (Goursat PDE cost scales with path length²)
- **Augmentation:** include time and lead-lag transformation for universal characteristic property
- **Batch resampling:** resample paths each gradient step to reduce variance (for Step 6 only)

**Deliverable:** `calibration/sig_mmd.py` — the central analytical tool of the thesis

### 3.4 Neural Surrogate Calibration (stretch goal) — Week 9

For models with closed-form Fourier/Laplace pricing (Heston, Quintic OU, PD-Bergomi) the calibration loop is already fast — gradient-based IV RMSE minimisation takes seconds. For the MC-only models (4F-PDV, rBergomi, Grey Bergomi), each pricing call requires a full MC simulation (~1–10 sec), making 50-step optimisation loops expensive. A neural surrogate replaces the MC pricer with a differentiable neural network approximation, reducing online calibration to milliseconds.

**The two-step deep calibration pipeline (Bayer, Horvath et al. 2019):**

**Step 1 — Offline training (expensive, done once):**
1. Sample $M = 10^4$–$10^5$ parameter vectors $\theta^{(i)}$ uniformly from the prior $\Theta$
2. For each $\theta^{(i)}$, run the MC pricer to compute the implied volatility surface $\Sigma^{(i)} \in \mathbb{R}^{n_K \times n_T}$ (e.g., 5 strikes × 6 maturities = 30-dimensional output)
3. Train a feed-forward network $\hat{f}_\phi: \Theta \to \mathbb{R}^{n_K \times n_T}$ to minimise $\sum_i \|\hat{f}_\phi(\theta^{(i)}) - \Sigma^{(i)}\|^2$

**Step 2 — Online calibration (fast, per date):**
Given observed market surface $\Sigma^*$:
$$\hat{\theta} = \arg\min_{\theta \in \Theta} \|\hat{f}_\phi(\theta) - \Sigma^*\|^2$$
Since $\hat{f}_\phi$ is a neural network, $\nabla_\theta \|\hat{f}_\phi(\theta) - \Sigma^*\|^2$ is computed via backpropagation in microseconds. Run `optax.adam` for 200–500 steps.

**Why this matters for rBergomi and Grey Bergomi in particular:**
- rBergomi has no closed-form pricing formula and no Markov property to exploit — MC is the only option
- The pricing map $\theta \mapsto \Sigma(\theta)$ is smooth (empirically confirmed by Bayer et al. 2019 for rBergomi), so a shallow MLP (3 hidden layers, 64 units each) fits it to IV RMSE < 0.1% with $M = 5 \times 10^4$ training samples
- Grey Bergomi has the same structure plus an additional parameter $\gamma$, requiring a slightly larger grid but the same approach

**Network architecture (suggested):**
- Input: $\theta \in \mathbb{R}^d$ ($d = 4$ for rBergomi: $H, \eta, \rho, \xi_0$; $d = 5$ for Grey Bergomi)
- Hidden: 3 × fully-connected layers, 64 units, SiLU activation
- Output: $\mathbb{R}^{n_K \times n_T}$ IV surface (flattened), no activation (raw IV values)
- Training: Adam, MSE loss, 100 epochs, batch size 256

**Purpose in thesis:** Speed benchmark and feasibility demonstration only. The primary calibration uses direct IV RMSE minimisation with MC repricing. The neural surrogate enables the dynamic calibration of Experiments 4–5 (250 daily dates × 8 models) to be tractable in wall-clock time.

**Deliverable:** `calibration/neural_surrogate.py`

### 3.5 Classifier Two-Sample Test (C2ST) — Week 8

A complementary diagnostic to sig-MMD. Train a binary classifier to discriminate paths from model A vs model B; test accuracy significantly above 50% implies the models are dynamically distinguishable.

**Method (Lopez-Paz & Oquab, NeurIPS 2017):**
1. Pool $N$ paths from model A (label 0) and $N$ paths from model B (label 1)
2. Train a simple LSTM or 1D-CNN classifier on 80% of the pooled data
3. Evaluate on the held-out 20%. Test accuracy = discrimination power.
4. Statistical significance via a binomial test: $H_0$: accuracy = 0.5.

**Why it complements sig-MMD:**
- Sig-MMD is a *kernel* test — it detects differences in the mean embedding. C2ST is effectively a *likelihood-ratio* test — it picks up whatever the classifier finds easiest to exploit, including non-linear structure sig-MMD might miss.
- C2ST provides an interpretable *discrimination accuracy* (e.g., "Heston vs rBergomi paths are 94% classifiable") alongside sig-MMD's p-value.
- The classifier's attention / gradient can reveal *which part of the path* (early vs late, price vs vol) drives distinguishability — sig-MMD gives a single scalar.

**Implementation:** ~50 lines with `equinox`. Use a 2-layer LSTM with hidden dim 32 → linear → sigmoid. Train with `optax.adam` for 200 steps on the same path arrays fed to sig-MMD.

**Deliverable:** `diagnostics/c2st.py`

### 3.6 Stylized Facts Scorecard — Week 8

A classical finance diagnostic: for each calibrated model (and for real SPX data in Experiment 6), compute a standardised table of stylized fact statistics. This is not novel, but every referee will expect it, and it provides interpretable low-dimensional projections of the holistic differences sig-MMD captures.

**Scorecard metrics:**

| Stylized fact | Statistic | What it tests |
|---|---|---|
| Vol clustering | $\text{ACF}(r_t^2)$ at lags 1, 5, 10, 20 | Long memory in variance |
| Leverage effect | $\text{CrossCorr}(r_t, \sigma_{t+h}^2)$ for $h = 1, \ldots, 10$ | Asymmetry direction + strength |
| Heavy tails | $\text{Kurt}(r_t)$ of daily log-returns | Tail fatness |
| Zumbach effect | $\text{Cov}(\sigma_t^2, R_{[t-\tau,t]}^2) - \text{Cov}(\sigma_t^2, R_{[t,t+\tau]}^2)$ | Time-reversal asymmetry |
| Vol-of-vol | $\text{Std}(\hat{\sigma}_t) / \mathbb{E}[\hat{\sigma}_t]$ (coefficient of variation of realised vol) | Second-order randomness |
| Skew decay | ATM skew $\psi(T) = \partial_k \sigma_{\text{IV}}(k,T)\big\|_{k=0}$ vs $T$ | Power-law $T^{H-1/2}$ for rough models |

**Implementation:** Pure NumPy/JAX statistics on simulated path arrays — no ML, no optimisation. ~80 lines.

**Output:** A table (models as rows, stylized facts as columns) with the market values in the final row. Colour-code: green = within 1 std of market, red = off. This table validates sig-MMD results: if sig-MMD says two models are distinguishable, the stylized facts should show *which* classical statistics differ.

**Deliverable:** `diagnostics/stylized_facts.py`

### 3.7 $p$-Variation Roughness Diagnostic — Week 8

A direct, interpretable, single-number summary of "how rough is this model's output?" Compute the empirical $p$-variation of simulated log-variance paths.

**Definition:** For a path $f: [0,T] \to \mathbb{R}$ sampled at times $0 = t_0 < t_1 < \cdots < t_L = T$:
$$V_p(f) = \sum_{i=1}^{L} |f(t_i) - f(t_{i-1})|^p$$

A process with Hölder regularity $\alpha$ has finite $p$-variation for $p > 1/\alpha$ and infinite $p$-variation for $p < 1/\alpha$. For rough Bergomi with $H \approx 0.1$, the critical $p \approx 1/H \approx 10$; for Heston ($H = 1/2$), the critical $p = 2$. Plotting $\log V_p$ vs $p$ for each model on the same axes produces a clean visualisation of the roughness spectrum.

**Implementation:** A loop over paths and $p$ values — ~20 lines. No ML, no PDE.

**Output:** A single figure showing $V_p$ curves for all 8 models + market realised vol paths (from Experiment 6 data). This is a natural Figure 1 for the Models chapter — it visually confirms the roughness hierarchy before any sig-MMD machinery is introduced.

**Deliverable:** `diagnostics/p_variation.py`

### 📚 Study Material — Signature-MMD Diagnostic

This is the intellectual core of the thesis. Invest the most study time here.

**1. Path signatures — from zero to working knowledge**
- **Lyons (2014)** "Rough paths, Signatures and the modelling of functions on streams." arXiv:1405.4537. A *survey* by the founder of rough paths theory, aimed at non-specialists. Read for the big picture: what signatures do, why they're universal, and how they connect to differential equations. Much more accessible than the 2007 monograph.
- **Chevyrev & Kormilitzin (2016)** "A Primer on the Signature Method in Machine Learning." arXiv:1603.03788. The best short tutorial. Read in full (~20 pages). Covers: the signature definition, its key properties (Chen's identity, shuffle product, universality), truncated signatures, and computational aspects. Work through the examples.
- **Kidger & Lyons (2021)** (arXiv:2001.00706), §2. Complement the above with the computational perspective — how signatures are actually computed in code.

**2. The signature kernel — from signatures to a kernel on paths**
- **Salvi, Cass, Foster, Lyons & Yang (2021)** (arXiv:2006.14794). *Read carefully.* This paper proves that the inner product of signatures (the "signature kernel") $k(\mathbf{x}, \mathbf{y}) = \langle S(\mathbf{x}), S(\mathbf{y}) \rangle$ satisfies the Goursat PDE:
$$\frac{\partial^2 k}{\partial s \partial t}(s,t) = \langle \dot{\mathbf{x}}_s, \dot{\mathbf{y}}_t \rangle \, k(s,t), \quad k(0,\cdot) = k(\cdot,0) = 1$$
This is the key computational trick: instead of computing infinite-dimensional signatures and taking their inner product, solve a 2D PDE. Read §2 (the kernel derivation), §3 (the Goursat PDE), and §4 (discretisation and convergence). Understand: the PDE is solved on a grid of size $L \times L$ where $L$ = path length, giving $O(L^2)$ cost per path pair.

- **Lemercier, Lyons & Salvi (2024)** (arXiv:2404.02926). The log-PDE improvement. Read §2 (why the standard Goursat PDE discretisation has first-order errors for rough paths) and §3 (the log-PDE fix). Relevant specifically for rough Bergomi paths.

**3. MMD with the signature kernel — the diagnostic metric**
- **Gretton et al. (2012)** (arXiv:0805.2368). Read *first* if you haven't already (listed in Project Overview study material). The generic MMD framework.
- **Chevyrev & Oberhauser (2022)** (arXiv:1810.10971), §3–4. The specialisation to signature kernels. Key results: (1) the signature kernel is *characteristic* (MMD = 0 iff laws are equal), (2) the two-sample test based on sig-MMD is consistent. Read the proof sketch of characteristicness — it follows from the fact that expected signatures determine laws (Theorem 1).
- **Salvi, Lemercier et al. (2021)** (arXiv:2109.03582). *Why sig-MMD beats standard MMD on paths.* Standard MMD with an RBF kernel on path-features (e.g., summary statistics) is blind to the *filtration* — it cannot distinguish processes with the same marginals but different conditional distributions. The higher-order KME (= signature kernel) captures the filtration. Read §2 (the counterexample showing standard MMD fails) and §3 (the higher-order KME construction).

**4. The permutation test — converting distances to p-values**
- **Alden, Horvath & Issa (2025)** (arXiv:2506.01718). Read §3 in detail. The permutation test works as follows:
  1. Pool the $N$ paths from model A and $N$ from model B into a single set of $2N$ paths.
  2. Compute the observed MMD² statistic.
  3. Randomly permute the $2N$ paths into two groups of $N$ each, $B = 1000$ times.
  4. For each permutation, compute the permuted MMD².
  5. The p-value = fraction of permutations where permuted MMD² ≥ observed MMD².
  Read §4 for Type II error analysis: when will the test *fail* to distinguish models that are actually different? The answer depends on $N$ (more paths = more power) and the "effect size" (how different the models actually are in path space).

**5. The lead-lag transformation — critical for characteristicness**
- **Chevyrev & Oberhauser (2022)**, §2.4. The lead-lag transform of a $d$-dimensional path $\mathbf{x}$ produces a $2d$-dimensional path by time-shifting: $\mathbf{x}^{LL}_t = (\mathbf{x}_{\lfloor t \rfloor}, \mathbf{x}_{\lceil t \rceil})$. This makes the signature sensitive to *quadratic variation* — without it, two processes with the same expected path but different volatilities could be indistinguishable. For stochastic volatility models, where the whole point is that models differ in their volatility dynamics, this is essential.
- **Salvi et al. (2021)** (arXiv:2006.14794), §5. Implementation details for lead-lag in the Goursat PDE framework.

**6. Sig-MMD as calibration loss (Step 6)**
- **Issa et al. (NeurIPS 2023)** (arXiv:2305.16274). Read §3 (sig-kernel scoring rule — proves strict properness: the minimum of the sig-MMD loss is attained iff the model distribution equals the target distribution) and §4 (Goursat PDE backpropagation — how to differentiate through the kernel computation w.r.t. model parameters). This is what makes sig-MMD calibration possible: `jax.grad` flows through `sigkerax` → through the Goursat PDE → through `diffrax` → to the model parameters $\theta$.
- **Briol et al. (2019)** (arXiv:1906.05944), §3. Proves that MMD-minimising parameter estimators are consistent and asymptotically normal under mild conditions. This is the theoretical guarantee that sig-MMD calibration produces valid parameter estimates.

**7. Classifier Two-Sample Test (C2ST)**
- **Lopez-Paz & Oquab (2017)** "Revisiting Classifier Two-Sample Tests." *ICLR 2017*. arXiv:1610.06545. *The defining paper.* Read §2 (the C2ST framework — train classifier, test accuracy > 50%), §3 (connection to divergence estimation), and §4 (comparison to MMD on benchmark datasets). Key insight: C2ST can detect differences that MMD misses when the distinguishing features are highly non-linear, because the classifier is trained to find *whatever* difference is easiest to exploit.
- **Friedman (2004)** "On multivariate goodness-of-fit and two-sample testing." Stanford Technical Report. An early proposal of using classifiers for two-sample testing. Read for the statistical perspective: why test accuracy is a valid test statistic.
- **Kim et al. (2021)** "Classification Accuracy as a Proxy for Two-Sample Testing." *Annals of Statistics* 49(1):411–434. Formalises the statistical properties of C2ST: proves that under mild conditions, C2ST is consistent (correctly rejects H₀ as $N \to \infty$ when distributions differ). Read §3 for the consistency theorem.

**8. Stylized facts of financial time series**
- **Cont (2001)** "Empirical properties of asset returns: stylized facts and statistical issues." *Quantitative Finance* 1(2):223–236. **The canonical reference.** Lists the 11 "stylized facts" of financial returns that any reasonable model should reproduce. Read the full paper — it's short (14 pages). Your scorecard (§3.6) directly tests whether each calibrated model matches these facts.
- **El Euch, Gatheral, Radoičić & Rosenbaum (2018)** (arXiv:1809.02098). For the Zumbach effect specifically — the time-reversal asymmetry. Already in the roadmap for Experiment 5; also relevant for the scorecard.

**9. $p$-Variation and roughness estimation**
- **Lyons (2014)** (arXiv:1405.4537), §2.1. Definition of $p$-variation and its relationship to Hölder regularity. The key result: a path with Hölder exponent $\alpha$ has finite $p$-variation for $p > 1/\alpha$.
- **Cass & Lyons (2015)** "Evolving communities with individual preferences." *Proc. London Math. Soc.* 110(1):83–107. §2 contains a clear treatment of $p$-variation for stochastic processes. Not essential reading but useful if you want the rigorous stochastic process version.
- **Cont & Das (2022)** (arXiv:2203.13820), §2. Their estimator of $H$ from realised vol is closely related to $p$-variation. Understanding the connection helps explain *why* your $p$-variation plot and Cont & Das's $H$ estimates might tell different stories.

---

## Phase 4: Experiments (Week 9–13)

> **Reframed experiment hierarchy:** Experiments 1 and 2 are setup/validation. **Experiments 3, 4, 5, and 6 are the thesis core** — they are the only ones that require sig-MMD and cannot be done with IV RMSE alone. Experiment 6 (sig-MMD on real market paths) is the single most novel contribution.

### Experiment 1: Synthetic Validation (Setup)

**Purpose:** Verify that the sig-MMD metric works as expected in controlled conditions, and probe the power of sig-MMD as a function of roughness.

> **Design note:** This experiment uses **theoretically motivated canonical parameters** for each model — not IV-calibrated parameters. The focus is on the diagnostic power of sig-MMD, not on calibration accuracy. Using calibrated parameters would conflate "do the models agree after fitting?" with "is sig-MMD a good test?" Published parameter choices: Heston ($\kappa=1, \theta=0.04, \xi=0.4, \rho=-0.7$), rBergomi ($H=0.1, \eta=1.9, \rho=-0.9$), Quintic OU (Abi Jaber et al. 2024 Table 1), PD-Bergomi (small $\varepsilon$, same $H, \eta, \rho$ as rBergomi), 4F-PDV (Gazzani & Guyon 2024 Table 2), Grey Bergomi (Jacquier et al. 2025), Sig-Vol (Alòs et al. 2025, truncation depth $m = 3$, coefficients from their Table 1 or retrained on rBergomi paths). Use flat $\xi_0(t) = 0.04$ throughout.

**Design:**
1. Fix canonical parameters for all 8 models (see note above); set rBergomi ground truth at $H = 0.1$
2. Simulate $M = 2048$ paths per model (RTX 3090 handles this comfortably; more paths = tighter permutation tests)
3. Compute pairwise sig-MMD between each model's paths and the rBergomi ground truth
4. **Key test:** Does sig-MMD correctly identify that path-dependent models (PD-Bergomi, 4F-PDV) are closer to rBergomi than purely Markovian models (Heston, Quintic OU)? Does Heston stand furthest away? Does Grey Bergomi cluster with or separate from plain rBergomi? Where does Sig-Vol land — does its signature basis give it an "unfair" proximity to rBergomi under sig-MMD?
5. Run the Alden et al. permutation test: can sig-MMD distinguish PD-Bergomi from rBergomi when their IV surfaces are indistinguishable?
6. Compute $p$-variation curves for all 8 models' simulated log-variance paths — verify the roughness hierarchy visually before any sig-MMD analysis
7. *(Sub-experiment A: roughness sensitivity)* Fix Heston and 1F Quintic OU at canonical parameters. Vary $H$ for rBergomi across $\{0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.49\}$ and compute $\text{sig-MMD}(\text{rBergomi}(H), \text{Heston})$ and $\text{sig-MMD}(\text{rBergomi}(H), \text{Quintic OU})$ for each $H$, with 1000 permutations per test and 10 independent bootstrap runs per $H$ value to produce confidence bands. Plot the two curves (with CIs) together. This directly answers *"does roughness matter, and how much?"*: at what $H^*$ does sig-MMD fail to distinguish rBergomi from a Markovian model? The 8-point grid is fine enough to pinpoint $H^*$ to within $\pm 0.025$. As $H \to 0.5$, rBergomi approaches a standard diffusion and the distances should collapse toward zero (or below permutation threshold).
8. *(Sub-experiment B: aBergomi convergence)* Simulate paths from the $n$-factor lifted Markov approximation (Abi Jaber & El Euch 2019) of rBergomi at $n = 1, 2, 4, 8, 16, 32$ with the rBergomi canonical parameters. Plot $\text{sig-MMD}^2(n\text{-Bergomi}, \text{rBergomi})$ vs $n$ with 10 bootstrap runs per $n$ for confidence bands. If sig-MMD is a valid distributional metric it must detect convergence. Secondary read-off: the critical $n^*$ at which the distance falls below permutation threshold answers *"how many factors does it take to be sig-MMD-indistinguishable from rough?"*

**Output:** Sig-MMD distance table (model vs. rBergomi ground truth), permutation test p-values, $p$-variation plot, roughness sensitivity curves (sig-MMD vs $H$), aBergomi convergence curve (sig-MMD$^2$ vs $n$)

### Experiment 2: Single-Date SPX Calibration (Setup)

**Data:** One date from CBOE (e.g., Oct 23, 2017 — same as Abi Jaber et al. papers).

**Design:**
1. Bootstrap $\xi_0(t)$ from the market variance swap curve
2. Calibrate each model via its best method (Fourier/AD for Heston, Quintic OU, PD-Bergomi; IV RMSE + MC for 4F-PDV, rBergomi, Grey Bergomi; linear regression on signature features for Sig-Vol)
3. Record IV RMSE per model — confirm all models achieve comparable static fit
4. For Quintic OU: also calibrate to VIX surface (joint calibration)
5. Compute the stylized facts scorecard (§3.6) for each calibrated model — vol clustering, leverage, kurtosis, Zumbach, vol-of-vol, skew decay

**Output:** IV surface fits, RMSE table, calibrated parameter table, stylized facts scorecard. This is a prerequisite for Experiment 3.

### Experiment 3: Cross-Model Signature Distance (CORE)

**This is the central novel experiment.** After calibrating all models to the same market date:

1. Simulate $N = 4096$ paths from each calibrated model (RTX 3090 kernel matrix: $4096^2 \times 4\text{B} = 64\text{MB}$ — fits easily in 24GB VRAM)
2. Compute the full $8 \times 8$ pairwise sig-MMD distance matrix, with 10 independent runs (resampling $N$ paths each time) to produce bootstrap confidence intervals on each distance
3. Run permutation tests (Alden et al., 1000 permutations) for all 28 model pairs; apply Romano-Wolf stepdown correction for multiple testing
4. Apply MDS or t-SNE to the distance matrix to produce a 2D "model map"
5. *(Sub-experiment)* For the Markovian models (Heston, 1F/2F Quintic OU), also compute FDM distances and verify agreement with sig-MMD
6. *(Sub-experiment)* Run C2ST (§3.5) on all 28 model pairs — report discrimination accuracy alongside sig-MMD p-values

**Key questions:**
- After static IV calibration, are models dynamically distinguishable?
- Does PD-Bergomi (with fitted $\varepsilon$) cluster with rBergomi or with the Markovian models?
- Does the 4F-PDV (path-dependent, no roughness) cluster with the Markovian models or with rBergomi? This directly tests: is it roughness or path-dependence that drives dynamic behaviour?
- Does the Quintic OU polynomial vol structure produce different path dynamics than the exponential vol of PD-Bergomi?
- **Grey Bergomi question:** Does Grey Bergomi's stochastic vol-of-vol produce detectably different paths from plain rBergomi? If $\gamma$ is small after calibration, this tests whether the additional structure is cosmetic or dynamically meaningful.
- **Sig-Vol circularity question:** Does sig-MMD rank the Sig-Vol model as closer to market than other models? If so, is this because the model is genuinely better, or because sig-MMD and Sig-Vol share the signature basis? Compare against C2ST ranking to test independence.
- Is there a clear Markovian / path-dependent / rough partition, or do models form a continuous spectrum?

**Output:** Heatmap of pairwise MMD², MDS embedding plot, permutation test p-value matrix, C2ST accuracy matrix, FDM vs sig-MMD comparison for Markovian models

### Experiment 4: Dynamic Calibration + Evolving Distances (CORE)

**Data:** Weekly calibrations over 2017–2022 (~250 dates), using either IV surfaces (OptionMetrics/WRDS) or daily historical paths (Yahoo Finance fallback).

**Design:**
For each date:
1. Calibrate all models (warm-start from previous date)
2. Compute the $8 \times 8$ sig-MMD distance matrix
3. Record calibrated parameters

**Analysis:**
- **Parameter stability:** Do Fourier/AD-calibrated parameters vary more or less smoothly than IV RMSE? (This was the original thesis question — it survives but is demoted from central to secondary.)
- **Evolving model distances:** Plot sig-MMD(PD-Bergomi, rBergomi) and sig-MMD(4F-PDV, rBergomi) over time. Does the gap shrink in calm markets and widen during crises?
- **Path-dependence vs roughness:** Does sig-MMD(4F-PDV, rBergomi) < sig-MMD(Heston, rBergomi) consistently? If so, path-dependence (even without roughness) brings models closer to rough dynamics.
- **PD-Bergomi interpolation:** How does $\varepsilon$ evolve over time when PD-Bergomi is re-calibrated weekly? Does it decrease during crises (model "becomes rougher")?
- **Optional — sig-MMD calibration comparison:** On a subset of dates, also calibrate via sig-MMD loss (§3.3 Step 6) and compare resulting parameters to Fourier/AD calibration.

**Output:** Time series of pairwise sig-MMD distances, parameter paths, convergence plots

### Experiment 5: Regime-Dependent Distinguishability (CORE)

Split the Experiment 4 time series into regimes (low vol 2017, COVID 2020, rate hikes 2022) and ask:

- In which regimes are rough and Markovian models sig-MMD-distinguishable? In which are they not?
- Does the Zumbach effect (time-reversal asymmetry of volatility, studied by El Euch, Gatheral et al. arXiv:1809.02098 under rough Heston) manifest differently across the model hierarchy? This is a direct test of dynamic path-space structure that IV RMSE cannot measure.
- **Connection to Cont & Das:** If roughness is an artefact of realised vol estimators (as Cont & Das 2022 suggest), then sig-MMD — which operates on simulated model paths, not realised vol — should still detect whatever structural difference the models encode. This provides independent evidence on the "rough or not" debate from an entirely different angle.

**Output:** Regime-stratified distance tables, box plots, formal test results per regime

### Experiment 6: Sig-MMD Against Real Market Paths (CORE — PRIORITY)

**This is the most direct test of model fidelity.** Instead of only comparing models against each other (Experiments 3–5), we compare each model's simulated paths against *observed* historical paths from the S&P 500.

**Data:** Rolling 30-day windows of $(t, \log S_t, \hat{\sigma}_t)$ from historical daily data (see §0.3 Option A). Use SPX daily closes + VIX (or 5-min realised vol from Oxford-Man Institute) as the volatility proxy. This gives ~200 historical path windows per year, covering 2011–2025.

**Design:**
1. Construct a set of $M$ historical market paths from rolling windows: $\mathbf{y}^{(j)} = (t, \log S_t, \text{VIX}_t)_{t \in [t_j, t_j+30]}$
2. For each calibrated model, simulate $N = 2048$ synthetic paths of the same dimension and horizon
3. Compute sig-MMD between each model's synthetic paths and the historical market paths
4. Run the Alden et al. permutation test: for each model, test H₀: "model paths and market paths are drawn from the same distribution"
5. Rank models by sig-MMD distance to market — this is the ultimate model-selection criterion
6. Compute the stylized facts scorecard (§3.6) for each model and for the real market data — compare side by side
7. Run C2ST (§3.5) between each model's paths and market paths — report "model vs market" discrimination accuracy
8. Compute $p$-variation curves for market realised vol paths and overlay on the model $p$-variation curves from Experiment 1

**Key questions:**
- Which model's simulated paths are closest to real SPX dynamics? Does a rough model win, or does a path-dependent Markovian model (4F-PDV) match reality equally well?
- Is the ranking model → market consistent with the model → model distances from Experiment 3? (If 4F-PDV is closest to rBergomi in Experiment 3 and both are close to market in Experiment 6, that strengthens the case for path-dependence over roughness.)
- Does the ranking change across market regimes? (Combine with Experiment 5 regime classification.)
- **The Cont & Das test, done properly:** Cont & Das argue roughness is an artefact of realised vol estimation. Sig-MMD on simulated vs. real paths side-steps the estimation issue entirely — it asks whether the *model's path dynamics* match the *market's path dynamics*, not whether an estimator of $H$ is biased.

**Practical considerations:**
- **VIX proxy caveat:** VIX ≠ instantaneous vol. For Quintic OU models, simulate model-implied VIX paths (using Prop. 2.4 of arXiv:2212.08297) for an apples-to-apples comparison. For other models, use MC-estimated 30-day integrated variance as a proxy.
- **Path count mismatch:** Historical data gives ~200 paths/year; models can generate unlimited paths. The permutation test handles unequal sample sizes, but power depends on the smaller sample.
- **Stationarity:** Rolling windows assume approximate stationarity within each regime. Split by regime (Experiment 5) to ensure this is reasonable.

**Output:** Model ranking by sig-MMD distance to market paths, permutation test p-values per model, regime-stratified rankings

### 📚 Study Material — Experiments

**1. Experiment 1 (Synthetic Validation) — understanding controlled tests**
- **Gretton et al. (2012)** (arXiv:0805.2368), §5. The power analysis section — how to determine the number of samples needed for the two-sample test to achieve a given power level. You'll need this to choose $N$ (number of paths per model) so the test has enough power to detect the differences you expect.
- **Romano & Wolf (2005)** "Exact and Approximate Stepdown Methods for Multiple Hypothesis Testing." *JASA* 100(469):94–108. When you run 28 pairwise tests (Experiment 3), you need to correct for multiple comparisons. Bonferroni is conservative; Holm-Bonferroni is better; Romano-Wolf is best (controls familywise error rate while maintaining power). Read §2 for the stepdown procedure.

**2. Experiment 2 (Single-Date SPX) — practical calibration references**
- **Abi Jaber & Li (2024)** (arXiv:2401.03345), Tables 1–2 and §4. This paper calibrates *all* the models you're studying (Heston, Quintic OU, rBergomi, PD-Bergomi) to the *same* SPX date (Oct 23, 2017). Their calibrated parameters are your starting points. Read the discussion of which models fit best in which regions of the surface (short vs. long maturity, ATM vs. wings).
- **De Marco & Henry-Labordère (2015)** "Calibration of Local Correlation Models to Basket Smiles." *J. Computational Finance* 18(3). Not directly relevant, but their §2 gives a clear exposition of why calibration is underdetermined (the inverse problem perspective) and why regularisation matters. Useful for the thesis write-up.

**3. Experiment 3 (Cross-Model Distance Matrix) — the key experiment**
- **Cox & Small (2014)** "Testing multivariate normality using kernel methods." *JASA* 109(508):1738–1748. Uses MMD as a goodness-of-fit test. Read for methodology: how to present MMD-based test results, how to interpret the heatmap, and how to report effect sizes alongside p-values.
- **Van der Maaten & Hinton (2008)** "Visualizing data using t-SNE." *JMLR* 9:2579–2605. t-SNE is one option for embedding the 8×8 distance matrix into 2D for visualisation. Read §2 for the algorithm and §3 for perplexity tuning. MDS (multidimensional scaling) is simpler and may be more appropriate for only 8 points — see Borg & Groenen (2005) *Modern Multidimensional Scaling*, Ch. 8.

**4. Experiments 4 & 5 (Dynamic + Regimes) — time-varying analysis**
- **El Euch, Gatheral, Radoičić & Rosenbaum (2018)** (arXiv:1809.02098). Read for the Zumbach effect: the empirical regularity that $\text{Cov}(\sigma_t^2, R_{[t-\tau,t]}^2) > \text{Cov}(\sigma_t^2, R_{[t,t+\tau]}^2)$ — past returns predict future vol more than future returns predict past vol. This is a *path-level* asymmetry that rough Heston can reproduce but standard Heston cannot. Experiment 5 should test whether sig-MMD detects this asymmetry.
- **Hamilton (2005)** "Regime switching models." In *The New Palgrave Dictionary of Economics*. For the regime classification in Experiment 5: how to formally define "low vol", "crisis", and "rate hike" regimes. A simple approach: use VIX level (< 15 = low vol, > 30 = crisis) or NBER recession dates.
- **Cont & Das (2022)** (arXiv:2203.13820), §5. The paper's discussion of *when* roughness matters — they suggest it should be most visible in high-frequency data during periods of rapid vol change. This connects to Experiment 5: if sig-MMD detects model differences primarily during crises, that's consistent with roughness being a high-frequency/stressed-market phenomenon.

---

## Phase 6: Analysis & Write-up (Week 13–16)

### Chapter Structure

| Chapter | Content | Key result |
|---|---|---|
| 1. Introduction | The "rough or not" debate; why IV calibration is insufficient; signature methods overview | Thesis question: does roughness matter in path space? |
| 2. Models | Full hierarchy: Heston → 1F/2F Quintic OU → PD-Bergomi → 4F-PDV → rBergomi → Grey Bergomi → Sig-Vol | Unified notation, role of each model in Markovian → path-dependent → rough → rough+ → signature-based spectrum |
| 3. Calibration & Diagnostic Methods | IV RMSE + Fourier/AD (primary calibration), Sig-MMD (diagnostic), C2ST, stylized facts scorecard, $p$-variation, FDM comparison | Sig-MMD as post-calibration diagnostic metric; complementary diagnostics validate and interpret |
| 4. Synthetic Validation | Experiment 1: sig-MMD correctly ranks model distances in controlled setting | Sig-MMD detects dynamic differences that IV RMSE misses |
| 5. Market Calibration & Model Distances | Experiments 2 & 3: single-date SPX, cross-model distance matrix | 8×8 heatmap, MDS embedding, permutation tests |
| 6. Models vs. Market: Real-Path Diagnostic | Experiment 6: sig-MMD between simulated model paths and observed historical paths | Model ranking by closeness to market dynamics |
| 7. Dynamic Distinguishability | Experiments 4 & 5: evolving distances, regime analysis | Time-varying Markovian/PD/rough boundaries, Cont & Das test |
| 8. Conclusion | What roughness means in path space, implications for model selection, future work | |

### Key Figures (minimum set)

1. IV surface fits: model vs market for all 8 models (one date)
2. Cross-model MMD heatmap (Experiment 3) — **the key thesis figure**
3. MDS/t-SNE embedding of the 8 models in path space
4. Permutation test p-value matrix (8×8)
5. **Model-vs-market sig-MMD bar chart** (Experiment 6) — ranking models by closeness to real SPX paths
6. Time series of sig-MMD(Markovian model, rBergomi) and sig-MMD(PD model, rBergomi) over 2017–2022
7. Regime-stratified box plots of model distances
8. FDM vs sig-MMD comparison scatter plot for Markovian models
9. $p$-variation curves for all 8 models + market realised vol (Models chapter — roughness hierarchy visualisation)
10. Roughness sensitivity: sig-MMD(rBergomi($H$), Heston) and sig-MMD(rBergomi($H$), Quintic OU) vs $H \in \{0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.49\}$ with bootstrap CIs — the "does roughness matter?" diagnostic power curve
11. aBergomi convergence curve: sig-MMD$^2(n\text{-Bergomi}, \text{rBergomi})$ vs $n$
12. Stylized facts scorecard table (models × facts, colour-coded vs market)
13. C2ST accuracy matrix (8×8, mirroring the sig-MMD heatmap)

---

## Critical Path & Dependencies

```
Week 1-2:   [Phase 0] Environment + project scaffold
                |
Week 2-4:   [Phase 1] Model implementations (parallelisable)
             |          |         |          |          |          |         |
           Heston   1F QOU  2F QOU  PD-Berg  4F-PDV  rBerg  gBerg  Sig-Vol
                |
Week 4-6:   [Phase 2] Calibration infrastructure
             |              |
          MC pricer    Fourier pricer
                |
Week 6-9:   [Phase 3] Calibration & diagnostics ← CRITICAL PATH
             |              |            |             |
           IV RMSE    Fourier/AD    Sig-MMD (+FDM)   C2ST / stylized facts / p-var
                |
Week 9-11:  [Phase 4] Experiments 1–3 (synthetic + single-date + cross-model)
                |
Week 11-12: [Phase 4] Experiment 6 (sig-MMD vs real market paths) ← PRIORITY
                |
Week 12-13: [Phase 4] Experiments 4–5 (dynamic + regime) ← CORE THESIS
                |
Week 13-16: [Phase 5] Write-up
```

**The single biggest risk:** the sig-MMD distance matrix (Experiment 3) not showing meaningful discrimination between calibrated models — i.e., all models appear sig-MMD-indistinguishable after IV calibration. This would be a *negative result*, but still publishable ("roughness doesn't matter in path space either"). Mitigate by running synthetic validation (Experiment 1) first to confirm sig-MMD can detect known differences.

---

## Technical Notes & Gotchas

### Forward Variance Curve $\xi_0(t)$
- This is an INPUT, not a parameter. Bootstrap from market variance swaps or SVI parameterisation.
- For Quintic OU: via $g_0(t) = \sqrt{\xi_0(t)/\mathbb{E}[p(X_t)^2]}$.
- For PD-Bergomi, rBergomi, and Grey Bergomi: $\xi_0(t)$ enters multiplicatively.
- For 4F-PDV: forward variance is implicit — the model self-initialises from the EWMA state.
- In synthetic experiments: use flat $\xi_0(t) = 0.04$ (20% vol).

### Signature Kernel Practical Tips
- The Goursat PDE discretisation in `sigkerax` uses a grid. Finer grid = more accurate but slower.
- **Lead-lag transformation** is critical for the kernel to be characteristic (distinguish distributions). Without it, the signature kernel is blind to quadratic variation.
- Start with `static_kernel_kind="rbf"` in sigkerax, then test `"linear"`.

### Rough Bergomi Simulation Pitfalls
- The hybrid scheme of Bennedsen et al. (2017) is essential — naive Cholesky doesn't scale.
- For $H < 0.1$, MC variance explodes. Use at least $10^5$ paths with variance reduction.
- McCrickerd & Pakkanen (2018) give a detailed treatment of antithetic + importance sampling tricks for rBergomi specifically.
- The `rbergomi` Python package (by Mikko Pakkanen) is a good reference implementation.

### Parameter Constraints
- Heston: Feller condition $2\kappa\theta > \xi^2$ (not strictly needed for simulation, but helps)
- Quintic OU: $\alpha_k \geq 0$ for odd $k$, fix $\alpha_5 = 1$
- PD-Bergomi: $\varepsilon > 0$ (strict), $H \in (0, 0.5)$, $\rho \in [-1, 0]$
- 4F-PDV: $h_R^{(i)}, h_V^{(i)} > 0$ (half-lives); $a_0 > 0$ (base vol); $\sigma_t^2 \geq 0$ enforced by the squaring
- Rough Bergomi: $H \in (0, 0.5)$
- Grey Bergomi: $H \in (0, 0.5)$, $\gamma \geq 0$ (grey parameter; $\gamma = 0$ reduces to rBergomi)
- Sig-Vol: $m \geq 1$ (truncation depth; typically $m = 3$ or $4$); $a_I \in \mathbb{R}$ (unconstrained — learned via linear regression)
- All models: $\rho \in [-1, 0]$ (leverage effect)
- Use `optax` with projected gradient or reparameterise (e.g., $\rho = -\text{sigmoid}(\tilde{\rho})$)

### Compute Budget Estimates
| Operation | RTX 3090 (est.) | Notes |
|---|---|---|
| Heston Fourier pricing (full surface) | < 0.1 sec | CPU-bound; GPU doesn't help |
| Quintic OU Fourier-Laplace (full surface) | ~0.1–0.5 sec | Riccati ODE solve on GPU |
| PD-Bergomi Fourier-Laplace (full surface) | ~0.5–2 sec | Depends on n-exponential truncation order |
| 4F-PDV MC (5×10⁵ paths, 252 steps) | ~5–8 sec | 6D SDE system; `vmap+jit` |
| Rough Bergomi MC (5×10⁵ paths, 252 steps) | ~10–20 sec | Hybrid scheme; FFT parallelises well on RTX 3090 |
| Grey Bergomi MC (5×10⁵ paths, 252 steps) | ~15–30 sec | Similar to rBergomi + grey BM covariance overhead |
| Sig-Vol calibration (linear regression on sig features) | ~1–5 sec | No MC optimisation loop; dominated by signature computation |
| Sig-Vol MC (5×10⁵ paths, 252 steps) | ~10–20 sec | Base process simulation + signature feature extraction |
| Sig-MMD (2048 paths, Goursat PDE, 252 steps) | ~5–15 sec | Kernel matrix 2048² = 16MB; PDE grid 252² = 64K pts |
| Sig-MMD (4096 paths, Goursat PDE, 252 steps) | ~15–40 sec | Kernel matrix 4096² = 64MB; used for Exp 3 |
| Sig-MMD bootstrap (10 runs × 2048 paths) | ~2–3 min | Confidence intervals on each pairwise distance |
| FDM (2048 paths, Markovian models) | ~1–3 sec | O(D) vs sig-MMD's O(D²); Markovian only |
| C2ST (4096 paths, LSTM 200 steps) | ~3–5 sec | Per model pair; train + eval |
| Stylized facts scorecard | < 0.1 sec | Pure NumPy/JAX statistics; negligible |
| $p$-variation curves | < 0.1 sec | Loop over $p$ values; negligible |
| Roughness sensitivity (8 $H$ values × 10 bootstrap × 2 models) | ~30–60 min | Sub-experiment A; parallelise over $H$ |
| aBergomi convergence (6 $n$ values × 10 bootstrap) | ~20–40 min | Sub-experiment B |
| Full calibration (50 optim steps, 5×10⁵ paths/step) | ~5–30 min per model | Dominated by MC resample per step |
| Dynamic calibration (250 dates × 8 models) | ~20–50 hours | Feasible on RTX 3090; 2–3 overnight runs |
| Experiment 6 (200 market paths × 8 models) | ~4–8 hours | Sig-MMD(model, market) × 10 bootstrap runs |

> **JIT warm-up:** First JAX call per function triggers XLA compilation (~10–60 sec). Subsequent calls use the compiled kernel. Always benchmark *after* warm-up.

### 📚 Study Material — Technical Topics

**1. Forward variance curve & SVI**
- **Gatheral & Jacquier (2014)** "Arbitrage-free SVI volatility surfaces." *Quantitative Finance* 14(1):59–71. arXiv:1204.0646. The definitive reference for SVI parameterisation and its arbitrage-free variant (SSVI). Read §2 (the SVI formula), §3 (calendar spread arbitrage — constraints on the forward variance curve), and §4 (the SSVI surface). You need this to bootstrap $\xi_0(t)$ from market IV data.
- **Roper (2010)** "Arbitrage Free Implied Volatility Surfaces." Working paper. More detail on no-butterfly and no-calendar-spread arbitrage constraints for IV surfaces.

**2. SDE simulation — numerical methods**
- **Kloeden & Platen (1992)** *Numerical Solution of Stochastic Differential Equations*, Springer, Ch. 9–10. The standard reference. Read §9.1 (Euler-Maruyama), §9.2 (Milstein — adds the $\frac{1}{2}\sigma\sigma'(\Delta W^2 - \Delta t)$ correction), and §10.2 (strong vs. weak convergence). For this thesis, weak convergence (convergence of distributions) is what matters — strong convergence (pathwise) is stronger than needed.
- **Jentzen & Kloeden (2011)** "Taylor approximations for stochastic partial differential equations." *CBMS-NSF Regional Conference Series* 83, SIAM. More modern treatment if you want to understand higher-order schemes.

**3. Variance reduction for Monte Carlo**
- **Glasserman (2003)**, Ch. 4 (antithetic variates) and Ch. 7 (control variates). The standard MC techniques. For rBergomi specifically, McCrickerd & Pakkanen (2018) adapt these to the fractional setting.
- **Jourdain & Kohatsu-Higa (2011)** "A review of recent results on approximation of solutions of stochastic differential equations." In *Stochastic Analysis with Financial Applications*, Springer. Read for the multilevel Monte Carlo (MLMC) perspective — potentially relevant for speeding up MC calibration across multiple discretisation levels.

**4. Kernel methods & RKHS — background for understanding sig-MMD**
- **Berlinet & Thomas-Agnan (2004)** *Reproducing Kernel Hilbert Spaces in Probability and Statistics*, Springer, Ch. 1–2. The mathematically rigorous treatment of RKHS. Read if you want to understand *why* the signature kernel being characteristic implies MMD = 0 ⟺ equal distributions.
- **Muandet, Fukumizu, Sriperumbudur & Schölkopf (2017)** "Kernel Mean Embedding of Distributions: A Review and Beyond." *Foundations and Trends in Machine Learning* 10(1-2):1–141. arXiv:1605.09522. The most comprehensive survey of kernel mean embeddings, MMD, and their applications. Read §2 (RKHS basics), §3 (kernel mean embeddings), and §4 (MMD). This fills the gap between Gretton et al. (2012) and the signature-specific papers.

**5. The Goursat PDE — numerical aspects**
- **Polyanin (2002)** *Handbook of Linear Partial Differential Equations*, Chapman & Hall, §7.3 ("The Goursat problem"). For the PDE theory background: existence, uniqueness, and classical numerical methods for the Goursat boundary-value problem. The sig kernel Goursat PDE is a special case with the product structure $f(s,t) = \langle \dot{\mathbf{x}}_s, \dot{\mathbf{y}}_t \rangle$.
- **Salvi et al. (2021)** (arXiv:2006.14794), §4. The specific discretisation scheme used in sigkerax: finite differences on the $(s,t)$ grid, with the `dyadic_order` parameter controlling refinement level.

---

## Drop Priority (if time runs short)

| Priority | What to drop | What survives |
|---|---|---|
| 1 (first) | Neural surrogate calibration | Everything else |
| 2 | Fine-tuning calibrated params (use published params directly for Exp 3) | Still get the distance matrix |
| 3 | Sig-Vol (Model 7) | 7 models still span full spectrum; sig-MMD circularity question deferred to future work |
| 4 | Grey Bergomi (Model 6) | 6 models still span full spectrum; grey BM adds nuance, not structure |
| 5 | C2ST sub-experiment (§3.5) | Sig-MMD p-values still answer the question; C2ST adds interpretability |
| 6 | Stylized facts scorecard (§3.6) | Sig-MMD + $p$-variation still cover distributional + roughness diagnostics |
| 7 | $p$-variation diagnostic (§3.7) | Roughness visible in Hurst estimation literature already; $p$-variation is visualisation only |
| 8 | FDM comparison sub-experiment | Sig-MMD covers all models; FDM is validation only |
| 9 | 4F-PDV (Guyon) | 5 models still span Markovian → PD → rough |
| 10 | 2F Quintic OU | Heston + 1F Quintic + PD-Bergomi + rBergomi |
| 11 | Dynamic experiments (4 & 5) | Exps 3 + 6 still give cross-model + market comparison |
| 12 | Sig-MMD as calibration loss (§3.3 Step 6) | Sig-MMD as diagnostic only |
| 13 (never) | Sig-MMD diagnostic pipeline + Exp 1–3 + Exp 6 + {Heston, 1F Quintic, PD-Bergomi, rBergomi} | **The irreducible core** |

> **Note on Experiment 6:** Sig-MMD on real market paths is now part of the irreducible core. It is the single most distinctive contribution — no prior work compares stochastic volatility models against real market paths in path-distributional space.

---

## Key References

| Short name | Full reference | Used for |
|---|---|---|
| **"Rough or not" debate** | | |
| Gatheral, Jaisson & Rosenbaum (2018) | "Volatility is rough." Gatheral, Jaisson & Rosenbaum. *Quantitative Finance* 18(6). arXiv:1410.3394 | **Foundational empirical paper** — H ≈ 0.1 from high-frequency realised volatility; launches the rough volatility programme |
| Cont & Das (2022) | "Rough volatility: fact or artefact?" Cont & Das. arXiv:2203.13820 | **Central counter-argument** — realised vol *always* appears rough (H < 0.5) even when instantaneous vol is diffusive; roughness may be artefact of estimator |
| Fukasawa, Takabatake & Westphal (2019) | "Is Volatility Rough?" Fukasawa, Takabatake & Westphal. arXiv:1905.04852 | Quasi-likelihood estimator confirms roughness but warns about estimation bias |
| Rosenbaum & Zhang (2022) | "On the universality of the volatility formation process: when machine learning and rough volatility agree." Rosenbaum & Zhang. arXiv:2206.14114 | Universal LSTM agrees with rough vol; universality argument for rough paradigm |
| Zarhali, Aubrun, Bacry, Bouchaud & Muzy (2025) | "Why is the volatility of single stocks so much rougher than the S&P500?" Zarhali et al. arXiv:2505.02678 | Nested factor model explains index vs stock roughness; dominant factor H≈0.11, residuals H≈0 |
| Gazzani & Guyon (2024) | "Pricing and calibration in the 4-factor PDV model." Gazzani & Guyon. arXiv:2406.02319 | **Model 4 (4F-PDV)** — time-homogeneous, Markovian, fits SPX without roughness; key alternative to rough vol |
| Guyon & Lekeufack (2023) | "Volatility is (mostly) path-dependent." Guyon & Lekeufack. *Quantitative Finance* 23(9). | **PDV foundation** — establishes that vol is well-explained by a functional of recent returns and realized vol; basis for 4F-PDV model |
| **Models** | | |
| Abi Jaber et al. (2024) | "Joint SPX-VIX calibration with Gaussian polynomial volatility models: deep pricing with quantization hints." Abi Jaber, Illand & Li. arXiv:2212.08297 | 1F Quintic OU, joint SPX-VIX, quantization |
| Abi Jaber & Li (2025) | "Capturing Smile Dynamics with the Quintic Volatility Model: SPX, Skew-Stickiness Ratio and VIX." Abi Jaber & Li. arXiv:2503.14158 | 2F Quintic OU, SSR, Zumbach |
| Abi Jaber & Li (2024) | "Volatility models in practice: Rough, Path-dependent or Markovian?" Abi Jaber & Li. arXiv:2401.03345 | SPX-only model comparison |
| Abi Jaber, Li & Lin (2024) | "Fourier-Laplace transforms in polynomial Ornstein-Uhlenbeck volatility models." Abi Jaber, Li & Lin. arXiv:2405.02170 | Fourier-Laplace for polynomial OU |
| Abi Jaber & El Euch (2019) | "Markovian structure of the Volterra Heston model." Abi Jaber & El Euch. *Statistics & Probability Letters*. arXiv:1908.09999 | Lifted Markov representation for Volterra processes — theoretical basis for PD-Bergomi simulation via sum-of-exponentials |
| Bayer, Friz & Gatheral (2016) | "Pricing under rough volatility." Bayer, Friz & Gatheral. *Quantitative Finance* 16(6) | Rough Bergomi definition & simulation |
| Bennedsen, Lunde & Pakk. (2017) | "Hybrid scheme for Brownian semistationary processes." Bennedsen, Lunde & Pakkanen. *Finance & Stochastics* 21 | Hybrid simulation scheme for fBm |
| Bergomi (2015) | *Stochastic Volatility Modeling.* Bergomi. CRC Press | Bergomi model framework, industry practice; PD-Bergomi generalises 1F/2F Bergomi |
| Jacquier, Oliveri Orioles & Zuric (2025) | "Rough Bergomi turns grey." Jacquier, Oliveri Orioles & Zuric. arXiv:2505.08623 | **Model 6 (Grey Bergomi)** — extends rBergomi with stochastic vol-of-vol via grey BM; achieves SPX/VIX joint calibration |
| Mura & Pagnini (2008) | "Characterizations and simulations of a class of stochastic processes to model anomalous diffusion." Mura & Pagnini. *J. Physics A* 41(28):285003 | Grey Brownian motion mathematical foundation — covariance structure, M-Wright function, simulation |
| Zhu, Loeper, Chen & Langrené (2020) | "Markovian approximation of the rough Bergomi model for Monte Carlo option pricing." *Mathematics* 9(5):528. arXiv:2007.02113 | Markovian multi-factor approximation of rBergomi for pricing — prior art on the "does roughness matter?" pricing question |
| **Signature methods** | | |
| Chevyrev & Oberhauser (2022) | "Signature moments to characterize laws of stochastic processes." Chevyrev & Oberhauser. *JMLR* 23(176). arXiv:1810.10971 | **Mathematical foundation of sig-MMD** — characterisation of process laws via sig moments; MMD metric on path space; non-parametric two-sample test |
| Salvi, Lemercier et al. (2021) | "Higher Order Kernel Mean Embeddings to Capture Filtrations of Stochastic Processes." Salvi, Lemercier, Liu, Horvath, Damoulas & Lyons. *NeurIPS 2021*. arXiv:2109.03582 | Why sig-MMD captures filtration information that standard MMD misses |
| Salvi et al. (2021) | "The Signature Kernel is the Solution of a Goursat PDE." Salvi, Cass, Foster, Lyons & Yang. *SIAM J. Math. Data Sci.* arXiv:2006.14794 | Goursat PDE derivation; efficient sig kernel computation |
| Lemercier, Lyons & Salvi (2024) | "Log-PDE Methods for Rough Signature Kernels." Lemercier, Lyons & Salvi. arXiv:2404.02926 | Higher-order sig kernel computation for rough paths (relevant for rBergomi) |
| Alden, Horvath & Issa (2025) | "Signature Maximum Mean Discrepancy Two-Sample Statistical Tests." Alden, Horvath & Issa. arXiv:2506.01718 | Formal sig-MMD two-sample test; Type II error analysis; permutation test — **core tool for Experiment 3** |
| **Issa, Horvath, Lemercier & Salvi (NeurIPS 2023)** | "Non-adversarial training of Neural SDEs with signature kernel scores." Issa, Horvath, Lemercier & Salvi. arXiv:2305.16274 | **Direct prior art** — sig-kernel scores for training neural SDEs; proves strict properness + Goursat PDE backprop; tests on rough vol simulation |
| Cuchiero et al. (2024) | "Joint calibration to SPX and VIX options with signature-based models." Cuchiero, Gazzani, Möller & Svaluto-Ferro. arXiv:2301.13235 | Signature-based models for SPX+VIX |
| Alòs et al. (2025) | "Volatility Modeling with Rough Paths: A Signature-Based Alternative to Classical Expansions." Alòs, Burés, de Santiago & Vives. arXiv:2507.23392 | **Model 7 (Sig-Vol)** — volatility as linear functional of truncated path signature; calibrated via linear regression; tests sig-MMD circularity |
| Pannier & Salvi (2024) | "A path-dependent PDE solver based on signature kernels." Pannier & Salvi. arXiv:2403.11738 | Sig kernels for solving path-dependent PDEs |
| **Calibration & training methods** | | |
| Lewis (2001) | *Option Valuation under Stochastic Volatility.* Lewis. Finance Press | Fourier pricing for Heston |
| Teng & Li (2025) | "Efficient Simulation and Calibration of the Rough Bergomi Model via Wasserstein Distance." Teng & Li. arXiv:2512.00448 | **Direct competitor** — Wasserstein-1 distributional calibration of rBergomi (operates on terminal distributions only, unlike sig-MMD); mSOE hybrid simulation scheme |
| Bayer, Horvath et al. (2019) | "On deep calibration of (rough) stochastic volatility models." Bayer, Horvath, Muguruza, Stemper & Tomas. arXiv:1908.08806 | Seminal two-step deep calibration: NN pricing map + gradient inversion |
| Briol et al. (2019) | "Statistical Inference for Generative Models with Maximum Mean Discrepancy." Briol, Barp, Duncan & Girolami. arXiv:1906.05944 | **Theoretical underpinning** — MMD estimators for intractable generative models: consistency, asymptotic normality, robustness to misspecification |
| Zhang et al. (2024) | "Efficient Training of Neural SDEs by Matching Finite Dimensional Distributions." Zhang, Viktorov, Jung & Pitler. arXiv:2410.03973 | FDM scoring rule: O(D) alternative to sig-kernel O(D²) by exploiting Markov property — scalability comparison for Markovian models |
| Lu & Sester (2024) | "Generative modelling of financial time series with structured noise and MMD-based signature learning." Lu & Sester. arXiv:2407.19848 | Sig-kernel MMD for S&P 500 financial time-series generation; close methodological overlap with this thesis |
| **Complementary diagnostics** | | |
| Lopez-Paz & Oquab (2017) | "Revisiting Classifier Two-Sample Tests." Lopez-Paz & Oquab. *ICLR 2017*. arXiv:1610.06545 | **C2ST framework** — train classifier to discriminate path distributions; accuracy > 50% = distinguishable; complements sig-MMD |
| Cont (2001) | "Empirical properties of asset returns: stylized facts and statistical issues." Cont. *Quantitative Finance* 1(2):223–236 | **Canonical stylized facts reference** — 11 empirical regularities of returns; basis for the scorecard in §3.6 |
| **Other** | | |
| El Euch, Gatheral et al. (2018) | "The Zumbach effect under rough Heston." El Euch, Gatheral, Radoičić & Rosenbaum. arXiv:1809.02098 | Zumbach effect (time-reversal asymmetry of vol) under rough vol — relevant to Experiment 5 regime analysis |
| McCrickerd & Pakkanen (2018) | "Turbocharging Monte Carlo pricing for the rough Bergomi model." McCrickerd & Pakkanen. *Quantitative Finance* 18(11). arXiv:1708.02563 | Variance reduction for rBergomi MC: antithetic + importance sampling; practical simulation reference |
| Tamayo-Rios, Schell & Alaifari (2025) | "Scalable Signature Kernel Computations for Long Time Series via Local Neumann Series Expansions." Tamayo-Rios, Schell & Alaifari. *NeurIPS 2025*. arXiv:2502.20392 | Superior Goursat PDE accuracy for rough paths + reduced memory + GPU scalability to very long series; potential sigkerax replacement |
| Wu, Ben Hammouda & Oosterlee (2025) | "SigMA: Path Signatures and Multi-head Attention for Learning Parameters in fBm-driven SDEs." Wu, Ben Hammouda & Oosterlee. arXiv:2512.15088 | Sig + multi-head attention for Hurst parameter estimation in rough SDEs; related to Experiment 4 |

---

## Future Directions

- **PD-Bergomi $\varepsilon$-sweep:** Run a systematic sweep of $\varepsilon$ values for PD-Bergomi and plot sig-MMD(PD-Bergomi($\varepsilon$), rBergomi) as a function of $\varepsilon$. This would trace the continuous interpolation between Markovian and rough in path space — a richer diagnostic than the discrete 8-model comparison. Closely related to the aBergomi convergence sub-experiment in Experiment 1 ($n \to \infty$ and $\varepsilon \to 0$ are two different routes to rBergomi in path space).
- **Grey Bergomi $\gamma$-sweep:** Analogously, sweep $\gamma$ for Grey Bergomi and plot sig-MMD(Grey Bergomi($\gamma$), rBergomi) to quantify the detectable effect of stochastic vol-of-vol in path space. Combined with the $\varepsilon$-sweep, this maps a 2D (roughness, vol-of-vol) parameter surface.
- **Path-dependent PDEs via signature kernels:** Pannier & Salvi (2024) show that signature kernels can solve path-dependent PDEs (PPDEs) numerically. A natural extension would use their PPDE solver to price options under non-Markovian models where Fourier methods are unavailable.
- **Higher-frequency market paths:** Experiment 6 uses daily data. Extending to intraday (5-min) paths from the Oxford-Man Realized Library would provide much richer path structure — roughness effects should be more pronounced at higher frequency (per Gatheral et al. 2018).
- **Realised vol vs VIX as $\hat{\sigma}_t$ proxy:** Experiment 6 uses VIX as a volatility proxy. An important robustness check is to replace VIX with realised volatility (e.g., 5-min RV) and test whether the model rankings change. If they do, the VIX proxy introduces systematic bias.
- **Multi-asset extension:** Apply the sig-MMD diagnostic framework to a basket of correlated assets, testing whether rough models provide better *cross-asset* dynamic dependence than Markovian alternatives.
- **Sig-MMD as calibration loss at scale:** Use Experiment 6's real market paths as the *target* in a sig-MMD calibration (§3.3 Step 6) — calibrating model parameters directly to match real path dynamics rather than IV surfaces. This is computationally expensive but conceptually the purest form of distributional calibration.
- **Adapted Wasserstein distance:** Backhoff-Veraart, Bartl, Beiglböck & Eder (2020) define an adapted (bicausal) Wasserstein distance that respects filtration structure — unlike standard Wasserstein, it penalises differences in conditional distributions, not just marginals. No GPU implementation currently exists, but if one appears it would be the natural filtration-aware complement to sig-MMD.
- **Signature-based volatility models — deeper investigation:** The Sig-Vol model (Alòs et al. 2025) is included at truncation depth $m \leq 4$ with linear regression calibration. Richer extensions include: Cuchiero et al. (2024) signature SDEs with non-linear dependence, higher truncation depths, and neural-network readout functions replacing linear regression. A systematic study of how truncation depth $m$ affects sig-MMD distance would quantify the representation power of signatures as a volatility model. The circularity question (does sig-MMD favour sig-based models?) deserves a dedicated paper with formal independence testing beyond C2ST.

---

## Future Work Queries

> Each query corresponds to one item in the Drop Priority table. If the item is dropped due to time constraints, the corresponding query is ready-made material for the thesis's "Future Work" chapter.

| Drop Priority | Dropped item | Future work query |
|---|---|---|
| 1 | Neural surrogate calibration | Can deep calibration (Bayer et al. 2019) of rough and path-dependent models be accelerated sufficiently via neural surrogates to enable real-time sig-MMD diagnostics on live IV surfaces? |
| 2 | Fine-tuning calibrated params | How sensitive is the sig-MMD distance matrix to the choice of calibration method — does fine-tuning from published parameters vs. fresh optimisation materially change the model topology? |
| 3 | Sig-Vol (Model 7) | Does sig-MMD exhibit diagnostic bias toward signature-based volatility models, or is the mathematical overlap between the signature kernel diagnostic and the signature model irrelevant to distributional testing? How does C2ST ranking compare? |
| 4 | Grey Bergomi (Model 6) | Is the stochastic vol-of-vol structure introduced by grey Brownian motion ($\gamma > 0$) detectable in path space via sig-MMD, and does it improve real-path fit beyond what roughness alone provides? |
| 5 | C2ST (§3.5) | Does the C2ST discrimination accuracy provide information beyond sig-MMD p-values — e.g., which path features (returns, vol, vol-of-vol) drive the classifier's decision, and do these align with the models' structural differences? |
| 6 | Stylized facts scorecard (§3.6) | Which of the Cont (2001) stylized facts are most informative for distinguishing stochastic volatility models after calibration, and does the scorecard ranking correlate with the sig-MMD ranking? |
| 7 | $p$-variation diagnostic (§3.7) | Can empirical $p$-variation of simulated log-volatility paths reliably distinguish rough ($H < 0.5$) from Markovian models, and at what sample size does the diagnostic become conclusive? |
| 8 | FDM comparison | For Markovian models, does the computationally cheaper FDM scoring rule (Zhang et al. 2024) agree with sig-MMD on relative model distances, and at what path dimensionality does the $O(D)$ vs $O(D^2)$ advantage become material? |
| 9 | 4F-PDV (Guyon) | Does path-dependence without roughness (the Guyon PDV mechanism) produce sig-MMD-detectable path dynamics that are genuinely distinct from both Markovian and rough models, or does it collapse toward one cluster? |
| 10 | 2F Quintic OU | Does the multi-factor Markovian structure of 2F Quintic OU produce path dynamics measurably different from 1F Quintic OU under sig-MMD, and does the second factor primarily affect smile dynamics or path-level behaviour? |
| 11 | Dynamic experiments (4 & 5) | How do pairwise sig-MMD distances between stochastic volatility models evolve across market regimes (calm, crisis, rate-hiking), and in which regimes does roughness become most or least detectable? |
| 12 | Sig-MMD as calibration loss (§3.3 Step 6) | Can sig-MMD serve as a practical calibration loss — not just a diagnostic — for stochastic volatility models, and does sig-MMD-calibrated parameter estimation yield different model rankings than IV RMSE calibration? |
