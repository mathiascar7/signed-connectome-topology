# Signed-Graph Topological Descriptors for Low-SNR rs-fMRI Connectome Classification

**A leakage-controlled hybrid architecture combining out-of-fold linear compression with non-Euclidean signed-graph invariants for ADHD-versus-control classification.**

> **Data Governance & Reproducibility.** Due to institutional IRB and clinical data-use agreements, raw subject-level diagnostic labels remain private. This repository provides the complete deterministic feature-extraction engine, the zero-leakage nested cross-validation pipeline, and an anonymized 20-subject schema sample (`data/sample_connectomes.csv`) to verify end-to-end execution. Full validation documentation is available to academic reviewers upon request.

---

## Abstract & Executive Summary

Resting-state functional MRI (rs-fMRI) classification in neurodevelopmental cohorts operates in an extreme high-dimensional, low-signal-to-noise regime ($p = 19,900$ edges, $n = 1,213$ subjects). Across cohorts, the mean absolute edge-weight shift between ADHD and neurotypical controls is **0.0021** (maximum single-edge shift **0.0154**), buried beneath a typical rs-fMRI physiological and scanner noise floor of $\approx 0.015$ ($\text{SNR} \approx 0.14$). Under strict leakage-free validation, standard tabular classifiers and unsigned graph baselines plateau at **0.60–0.63 ROC-AUC**.

Rather than deploying over-parameterized black-box architectures that memorize $p \gg n$ noise, we formulate connectome classification through **Structural Balance Theory** and **signed-graph walk dynamics**. By explicitly preserving negative functional correlations ($W_{ij} < 0$) and pairing an out-of-fold (OOF) L2-regularized linear probability anchor with 1,000 deterministic node-level signed-graph descriptors refined by heavily regularized gradient-boosted trees (LightGBM), our framework achieves **0.7164 ± 0.0185 mean outer-fold ROC-AUC** in **5×5 nested cross-validation**—an **+8.64 to +11.64 percentage-point gain** over conventional baselines without using demographic, behavioral, or anatomical atlas priors.

---

## 1. Problem Formulation: The Low-SNR & $p \gg n$ Barrier

- **Input Representation:** Each subject $m \in \{1, \dots, 1213\}$ is represented exclusively by a symmetric, zero-diagonal $200 \times 200$ whole-brain Pearson correlation matrix $W^{(m)} = (W^{(m)})^\top$, flattened into its strict upper triangle of $p = \binom{200}{2} = 19,900$ continuous edge features (`0throw_1thcolumn` through `198throw_199thcolumn`).
- **Blind Experimental Protocol:** All demographic covariates (`age`, `sex`, `site`, `site_id`), clinical questionnaires, and anatomical parcellation labels are excluded prior to modeling. ROI indices $\{0, \dots, 199\}$ are treated strictly as anonymous graph vertices.
- **Statistical Bottleneck:** Empirical cohort-level edge differences exhibit a mean absolute deviation of $0.0021$ and a maximum single-edge deviation of $0.0154$, matching the expected thermal and physiological noise floor of resting-state BOLD correlations.

![Mean control and ADHD connectomes and overlapping edge-weight distributions](assets/figures/02_page03_mean_connectomes_overlap.jpg)

*Figure 1. Empirical cohort-averaged connectomes (Panels A–B) and overlapping edge-weight density distributions (Panel C) illustrating the $\Delta\mu = 0.0021$ mean shift and $\max \vert{}\Delta r\vert{} = 0.0154$ ceiling that define the sub-noise classification regime.*

---

## 2. Theoretical Motivation & Engineering Matrix

Conventional connectomic pipelines routinely threshold correlation matrices to positive weights ($W^+ = \max(W, 0)$) or take absolute values ($\vert{}W\vert{}$). Mathematically, both operations destroy **triadic sign consistency** ($W_{ij}W_{jk}W_{ik} < 0$) and **local antagonistic coupling**, discarding **75.6%** of the discriminative topological Information Gain recovered by our model.

| Bottleneck in rs-fMRI Modeling | Failure Mode of Conventional Pipelines | Engineered Architectural Solution |
| :--- | :--- | :--- |
| **Severe Dimensionality ($p = 19,900 \gg n = 1,213$)** | Unconstrained splits and dense neural layers overfit spurious edge fluctuations. | Fold-contained two-sample Kolmogorov–Smirnov (KS) filtering (`top_k=500` raw edges; `top_k=300` graph descriptors) paired with strong L2/L1 regularization. |
| **Sub-Noise Edge Shifts ($\Delta\mu \approx 0.0021$)** | Single-edge statistics lack statistical power; whole-brain global averages wash out localized circuit alterations. | Node-level topological aggregation mapping 19,900 noisy edges into 1,000 deterministic structural-balance and greedy-walk invariants (5 per ROI). |
| **Loss of Anti-Correlations ($W_{ij} < 0$)** | Positive-only thresholding and $\vert{}W\vert{}$ transforms erase functional segregation and triadic frustration. | Explicit decomposition into $W^+ = \max(W, 0)$ and $W^- = \min(W, 0)$, quantifying triadic frustration and negative-strength concentration. |
| **Global Linear vs. Local Non-Linear Geometry** | Linear models miss threshold interactions; decision trees poorly approximate dense linear combinations. | Two-stage nested stacking: an OOF L2 logistic regression probability anchor supplies global linear risk to LightGBM alongside localized graph invariants. |

---

## 3. Mathematical Definition of Signed-Graph Descriptors

For each subject matrix $W \in \mathbb{R}^{200 \times 200}$ ($W_{ii} = 0$), let $W^+ = \max(W, 0)$, $W^- = \min(W, 0)$, and positive node strength $s_i^+ = \sum_{j=1}^{200} W_{ij}^+$. Five deterministic invariants are computed per vertex $i \in \{0, \dots, 199\}$ without access to target labels $y$:

| Descriptor | Mathematical Formulation | Structural Property Captured |
| :--- | :--- | :--- |
| **`Region_i_Frustration`** | $\displaystyle F_i = \frac{1}{2} \sum_{\substack{j,k \\ T_{jk}^{(i)} < 0}} \lvert T_{jk}^{(i)} \rvert, \quad T_{jk}^{(i)} = W_{ij}W_{jk}W_{ik}$ | **Signed Triadic Imbalance:** Quantifies the cumulative magnitude of frustrated closed triangles incident to node $i$ under Heider–Harary structural balance theory. |
| **`Region_i_Antagonism`** | $\displaystyle A_i = \sum_{m=1}^{5} \lvert w_{i,(m)}^- \rvert, \quad w_{i,(1)}^- \le \dots \le w_{i,(d_i^-)}^- < 0$ | **Peak Negative Coupling:** Sum of the $k=5$ strongest negative correlations incident to node $i$, isolating focal antagonistic coupling. |
| **`Region_i_Inhibitory`** | $\displaystyle I_i = \sum_{j=1}^{200} \lvert W_{ij}^- \rvert$ | **Total Negative Strength:** Cumulative magnitude of all negative functional correlations incident to vertex $i$. |
| **`Region_i_Deceleration`** | Let $\mathcal{H} = \{v : s_v^+ \ge P_{95}(s^+)\}$ and $\widetilde{W}^+$ be $W^+$ with rows/columns in $\mathcal{H}$ zeroed. For $i \notin \mathcal{H}$, follow greedy non-backtracking max-weight steps $v_0=i \to v_1 \to \dots \to v_L$ ($L \le 200$ or first cycle): $\displaystyle D_i = \sum_{t=1}^{L} \max\left(0, \widetilde{W}_{v_{t-1}v_t}^+ - \widetilde{W}_{v_t v_{t+1}}^+\right)$ | **Peripheral Weight Attenuation:** Measures cumulative correlation decay along greedy positive pathways once top-5% high-strength hubs are bypassed ($D_i = 0$ for $i \in \mathcal{H}$). |
| **`Region_i_DeadEndCrash`** | Starting at $v_0 = i$, iteratively transition to $v_{t+1} = \arg\max_{u \notin \{v_0,\dots,v_t\}} W_{v_t u}^+$ while $\max_u W_{v_t u}^+ > 0$, terminating at $v_{\text{term}}$: $\displaystyle C_i = s_{v_{\text{term}}}^+$ | **Self-Avoiding Walk Basin Strength:** Evaluates the positive node strength of the terminal vertex reached by a self-avoiding greedy traversal on $W^+$. |

> **Methodological Scope:** Because $W$ is a symmetric, undirected Pearson correlation matrix over anonymous ROIs, these invariants characterize static topological balance, negative-correlation concentration, and weight-decay structure on correlation weights—not directed causal signaling or anatomical pathways.

---

## 4. Zero-Leakage 5×5 Nested Cross-Validation Architecture

In high-dimensional neuroimaging ($p \gg n$), performing feature selection, scaling, or hyperparameter tuning across the full dataset introduces severe optimistic bias. Our evaluation enforces strict separation across every nested boundary:

1. **Outer Loop (`5-fold StratifiedKFold`):** Partitions subjects into 80% outer training and 20% untouched outer test folds. No test-fold sample influences feature ranking, scaling, linear meta-feature weights, or tree hyperparameters.
2. **Inner Loop (`5-fold StratifiedKFold`):** Within each outer training set, `KSTopFeatureSelector` (`top_k=500` raw edges; `top_k=300` topological descriptors) and `StandardScaler` are fitted exclusively on each inner training split.
3. **Leakage-Free Stacking (`LINEAR_META_RISK_SCORE`):** Inside each inner training fold, an internal 3-fold `cross_val_predict` generates strictly out-of-fold probability estimates from an L2-penalized logistic regression ($C = 0.01$), preventing target leakage into LightGBM.
4. **Constrained Bayesian Optimization (`Optuna TPE`, 40 trials/fold):** Tunes shallow, heavily regularized LightGBM trees (`max_depth` $\in [3, 7]$, `num_leaves` $\in [8, 32]$, `min_child_samples` $\in [15, 60]$, `reg_alpha`, `reg_lambda` $\in [1.0, 20.0]$, `path_smooth` $\in [0.0, 15.0]$) solely on mean inner-validation ROC-AUC.

```mermaid
flowchart TD
    A["Full Labeled Cohort (n = 1,213, p = 19,900)"] --> B["Outer 5-Fold StratifiedKFold"]
    B --> C["Outer Train Split (80%)"]
    B --> H["Held-Out Outer Test Fold (20%)"]
    C --> D["Inner 5-Fold StratifiedKFold"]
    D --> E["3-Fold Internal OOF Logistic Anchor (top_k=500 edges, L2 C=0.01)"]
    D --> F["KS Selector (top_k=300 Signed-Graph Features) on Inner Train"]
    E --> G["Optuna TPE (40 Trials): Maximize Mean Inner-Val ROC-AUC"]
    F --> G
    C --> I["5-Fold OOF Logistic Anchor + KS top_k=300 Fitted on Full Outer Train"]
    G --> M["Train Best Regularized LightGBM on 301 Outer-Train Features"]
    I --> M
    M --> J["Single Unbiased Evaluation on Held-Out Outer Test Fold"]
    H --> J
