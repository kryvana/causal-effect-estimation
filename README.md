# Barycenter Approximating of Causal Effects

Code for my bachelor's thesis *Barycenter Approximating of Causal Effects* (Technical University of Munich, 2026).

**Author:** Anastasiia Kryvoruchko
**Supervisor:** Prof. Dr. Seyed Jalal Etesami, Chair of Decision Analytics, TUM School of Computation, Information and Technology

The full thesis is in [`thesis.pdf`](thesis.pdf).

## Overview

Causal effects are normally identified and estimated from a single, known causal graph. When the graph is learned from data, for example by causal discovery, its structure is uncertain: each edge exists only with some probability. The effect P(Z | do(X = x)) may then be identifiable in some versions of the graph and not in others, and different versions may need different estimation strategies.

This project approximates the causal effect in such a probabilistic semi-Markovian graph with a Wasserstein barycenter:

1. **Generate graph and data (DGP).**
   - **Directed edges** are drawn with Bernoulli parameter `q`. The graph is stored as a strictly lower-triangular adjacency matrix, so it is always acyclic.
   - **Bidirected edges** are hidden confounders `U0, U1, …`, each pointing to two observed variables. They are drawn with Bernoulli parameter `p`. Each pair of observed variables gets two draws, so a pair is confounded with probability 1 − (1 − p)².
   - **Edge probabilities:** every edge gets an existence probability drawn from U(0, 1).
   - **Data:** sampled from a structural equation model on the graph (see [Models](#models)).
2. **Split the graph** into all 2^m subgraphs (surrogates), one for each way of keeping or dropping the m uncertain edges.
3. **Identify and estimate** the effect in each subgraph with [DoWhy](https://github.com/py-why/dowhy). Depending on what the subgraph allows, this uses the back-door criterion, the front-door criterion or an instrumental variable.
4. **Aggregate** the interventional distributions of the identifiable subgraphs into a weighted Wasserstein barycenter with [POT](https://pythonot.github.io/). The cost matrix is the squared Euclidean distance between grid points, normalised by its largest entry.
5. **Evaluate** the barycenter against the true interventional distribution with the 2-Wasserstein distance. The true distribution is computed on the full dataset, including the hidden variables.

## Models

The thesis evaluates six configurations, M1–M6. Each notebook implements one data-generating model:

| Thesis model | Notebook | Structural equation for Xᵢ | Noise εᵢ |
|---|---|---|---|
| M1, M2, M3 (linear) | `notebooks/01_linear_gaussian.ipynb` | Σ βⱼᵢ Xⱼ + εᵢ | N(0, 1) |
| M4 (binary) | `notebooks/02_binary.ipynb` | (⊕ Xⱼ) ⊕ εᵢ, the XOR of the parents | Bernoulli(0.1) |
| M5 (polynomial) | `notebooks/03_nonlinear_quadratic.ipynb` | Σ (αⱼᵢ Xⱼ² + βⱼᵢ Xⱼ) + εᵢ | ½ N(−2, 0.5²) + ½ N(2, 0.5²) |
| M6 (tanh) | `notebooks/04_nonlinear_tanh.ipynb` | tanh(Σ βⱼᵢ Xⱼ) + εᵢ | ½ N(−2, 0.5²) + ½ N(2, 0.5²) |

Sums and XORs run over the parents Xⱼ of Xᵢ, and variables without parents are pure noise. Coefficients are drawn from U(−1, 1). In M5 and M6, each variable is standardised after it is generated. The observed variables are `X`, `W`, `V` and `Z`, with `X` as treatment and `Z` as outcome.

How each subgraph's interventional distribution is formed:

| Model | Target | Distribution of a subgraph |
|---|---|---|
| Linear | do(X = x) for x = −4, …, 4 | The estimate ATE · x, represented as a discretised Gaussian of fixed width centred on it, on a 100-point grid |
| Binary | P(Z = 1 \| do(X = 1)) | Two-point distribution on {0, 1}, using P(Z = 1 \| do(X = 1)) = (1 + ATE) / 2 for the XOR model |
| Non-linear | P(Z \| do(X = 2)) | Histogram (100 bins) of interventional outcomes simulated with gradient boosting (back-door, front-door) or DML-IV from [EconML](https://github.com/py-why/EconML) (instrumental variable) |

## Weighting strategies

Before the barycenter is computed, each identifiable subgraph gets a weight, and the weights are normalised to sum to 1. The thesis compares three strategies:

| Strategy | Thesis model | Weight of a subgraph |
|---|---|---|
| Probability of the subgraph | M1 | Product over the uncertain edges of pₑ for each kept edge and 1 − pₑ for each dropped edge (`"prob of subgraph"`) |
| Random | M2 | Uniform random number in [0, 1) |
| Inverse conditional entropy | M3, M4, M5, M6 | 1 / exp(H), where H is the entropy of the residual of `Z` after predicting `Z` from its observed parents in that subgraph (`"weightEnt"` in notebook 01, `"weight"` in 02–04) |

Inverse conditional entropy gives larger weights to subgraphs in which the parents of `Z` predict `Z` well. The residual and its entropy are computed differently for each model:

| Model | Residual of `Z` | Entropy |
|---|---|---|
| Linear | Linear regression | Differential entropy |
| Binary | `Z` XOR a decision tree's prediction | Shannon entropy |
| Non-linear | Gradient boosting | Differential entropy |

The notebooks use inverse conditional entropy, as the thesis does from M3 on. In the thesis, M1–M3 differ only in their weights. To reproduce M1 or M2 with notebook 01, open the estimation cell of the Linear Model (the one that builds `map_of_estimates_and_weights`) and replace `subgraph["weightEnt"]` with:
- **M1:** `subgraph["prob of subgraph"]`
- **M2:** `np.random.rand()`

The same change works in notebooks 02–04, where the entropy weight is `subgraph["weight"]`.

### Linear Model 1

Notebook 01 also contains a second linear model, "Linear Model 1", titled "Self-Defined Weights" (variables `map_adj_1`, `map_dataset_1`, `map_of_subgraphs_1`). Its graph is drawn the same way but without edge probabilities. Every edge is equally uncertain, so subgraph probabilities are not defined. It was set up for weights that do not depend on edge probabilities, such as random ones. In this notebook it uses inverse-entropy weights.

## Key results

From the thesis:
- **Overall accuracy:** in the representative single panels (Figures 18–23), the barycenter's 2-Wasserstein error is below 5% of the grid width for all six models.
- **Scattered estimates:** the barycenter stays close to the true distribution even when individual subgraph estimates are far off. In the binary model M4, a single surrogate has an error of 0.405, while the barycenter has 0.047.
- **Weighting:** inverse conditional entropy gave the lowest error (M3: 0.0224), ahead of random weights (M2: 0.0227) and subgraph probability (M1: 0.0255). The differences are small.
- **Scope:** the experiments used 5–7 observed variables, 5,000–15,000 samples and one random seed.

## Setup

The notebooks were run with Python 3.10. With [conda](https://docs.conda.io):

```bash
conda create -n barycenter-causal-effects python=3.10
conda activate barycenter-causal-effects
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Without conda, using Python's built-in virtual environments:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open a notebook in `notebooks/`. The notebooks were saved with a kernel called "Python (Thesis)", which exists only on the author's machine. If JupyterLab asks you to select a kernel, choose **Python 3 (ipykernel)**. Then run **Kernel → Restart Kernel and Run All Cells**.

### What to expect when running

- **Run time.** The number of subgraphs grows as 2^m in the number of uncertain edges, so dense settings (high `q` and `p`) take noticeably longer to run.
- **Skipped subgraphs.** A subgraph is left out of the barycenter when its effect cannot be estimated. Notebooks 02–04 print one `Skipping subgraph q=…, p=…: <reason>` line for each, so their estimation cells print several thousand lines. Notebook 01 skips the same cases silently. The common reasons are:
  - `'NoneType' object has no attribute 'get'`: the subgraph has no directed path from `X` to `Z`. DoWhy treats the effect as zero and returns no estimand.
  - `None of [Index(['U0'], …)] are in the [columns]`: the adjustment set DoWhy found contains a hidden variable `U`, which is not in the observed data.
  - `Only singleton frontdoor variables are supported…` (binary model only): DoWhy's front-door estimator handles a single mediator only.
- **DoWhy warnings.** `UserWarning: … variables are assumed unobserved because they are not in the dataset` is expected; these are the hidden `U` variables.
- **Reproducibility.** Graphs and data come from a fixed seed (`seed = 40`), so running a notebook from the top gives the same results every time. Re-running single cells continues the random sequence and changes the numbers. One exception: the front-door estimates in notebooks 03 and 04 resample residuals with an unseeded `np.random.default_rng()`, so they change slightly from run to run.
- **Relation to the thesis figures.** The notebooks run a grid of `q` and `p` values on 4 observed variables. The figures in the thesis come from separate experiment runs with 5–7 observed variables and the parameters given in each caption, so the numbers differ.

## Repository structure

```
notebooks/          one notebook per data-generating model
requirements.txt    Python dependencies
thesis.pdf          the thesis
```

The experiment scripts, which vary one parameter at a time (sample size, intervention value, `p`, `q`), will be added later.

## Citation

```bibtex
@mastersthesis{kryvoruchko2026barycenter,
  author = {Kryvoruchko, Anastasiia},
  title  = {Barycenter Approximating of Causal Effects},
  type   = {Bachelor's thesis},
  school = {Technical University of Munich},
  year   = {2026}
}
```

## License

The code is released under the MIT License, see [LICENSE](LICENSE). The thesis text (`thesis.pdf`) is not covered by this license.
