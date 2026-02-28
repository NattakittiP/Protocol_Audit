# Protocol-Level Audit Runner

This repository provides a **single, locked experiment runner** that reproduces the **protocol-level auditing study**:

> **Protocol-Level Auditing of Machine Learning Evaluation: Disentangling Split Policy, Leakage, and Decision Stability**

The runner executes **Phase 1–3 end-to-end**, enumerates the full audit matrix, and exports **fold-level metrics**, **configuration summaries**, **winners**, **winner flip percentages**, **OOF predictions**, and **leakage artifacts** in a **fully reproducible** manner.

---

## Table of Contents

- [Key Idea](#key-idea)
- [What This Runner Audits](#what-this-runner-audits)
- [Repository Contract](#repository-contract)
- [Datasets](#datasets)
  - [Dataset A — ICU Mortality Cohort (Real Data)](#dataset-a--icu-mortality-cohort-real-data)
  - [Dataset B — Synthetic Physiologic Cohort](#dataset-b--synthetic-physiologic-cohort)
- [Experimental Design](#experimental-design)
  - [Split Policies (S1, S2)](#split-policies-s1-s2)
  - [Leakage Protocols (P0–P3)](#leakage-protocols-p0p3)
  - [Models and Hyperparameters](#models-and-hyperparameters)
  - [Calibration Strategy](#calibration-strategy)
  - [Metrics and Winner Rule](#metrics-and-winner-rule)
  - [Winner Flip Percentage (Phase 2)](#winner-flip-percentage-phase-2)
- [Phases](#phases)
  - [Phase 1 — Main Audit Matrix (Dataset A)](#phase-1--main-audit-matrix-dataset-a)
  - [Phase 3 — Synthetic Control (Dataset B)](#phase-3--synthetic-control-dataset-b)
  - [Phase 2 — Seed Reproducibility (Dataset A, P0 only)](#phase-2--seed-reproducibility-dataset-a-p0-only)
- [Outputs](#outputs)
- [How to Run](#how-to-run)
- [Design Notes / Guarantees](#design-notes--guarantees)
- [Citation](#citation)

---

## Key Idea

Machine learning evaluation is often called “robust” when cross-validation results are stable across random seeds.  
This work argues (and audits) a stronger claim:

**Seed stability measures stochastic variability — it does not certify protocol validity.**

We therefore audit evaluation protocols at the **protocol level**, explicitly separating:

- **Split policy** effects (stratified vs. group-based CV),
- **Leakage protocol** effects (fold-safe vs. progressively violated variants),
- **Model family** effects (linear vs. tree-based),
- **Decision stability** via winner identity and **winner flip percentage**.

---

## What This Runner Audits

The script `jcsse_audit_runner.py` implements a protocol-level audit of ML evaluation across:

- **Split policy**: `S1` (StratifiedKFold) vs `S2` (GroupKFold)
- **Leakage protocol**: `P0–P3` (strictly controlled → progressively violated)
- **Model family**: 5 classical models
- **Decision stability**:
  - winners selected by a **fixed lexicographic rule** *(AUROC → AP → Brier)*
  - seed robustness measured by **winner flip percentage** over **20 seeds**

---

## Repository Contract

This repository is intentionally **not** a general-purpose training framework.

It is an **executable specification** of the protocol used in the paper:

- Seeds are fixed
- Protocol definitions are fixed
- Grids are fixed
- Outputs are fixed

A single command should reproduce the reported audit results (given the same data files and environment).

---

## Datasets

The runner expects **two CSV files** in the working directory.

### Dataset A — ICU Mortality Cohort (Real Data)

- **File**: `full_analytic_dataset_mortality_all_admissions.csv`
- **Label column**: `label_mortality` (binary in-hospital mortality)
- **Group column**: `subject_id` (used for group-based splitting, `S2`)
- **Dropped columns** (not used as features): `hadm_id`, `label_mortality`, `subject_id`
- **Use case**: protocol audit under `S1` and `S2`

### Dataset B — Synthetic Physiologic Cohort

- **File**: `Synthetic_Dataset_1500_Patients_precise.csv`
- **Core column**: `TG4h` (4-hour triglycerides)
- **Binary label**: `TG4h >= global_p75(TG4h)` (top quartile threshold, computed globally)
- **Optional ID columns**: if present, one of:
  - `ID`, `Id`, `id`, `patient_id`, `PatientID`, `subject_id`
  - are detected and dropped
- **Synthetic missingness**:
  - **MCAR 15%**
  - applied to selected numeric-like features:
    - `Age, BMI, TG0h, HDL, LDL, Hematocrit, TotalProtein, WBV`
  - applied independently per outer fold **train/test** partitions
  - fixed RNG seed: `B_MISS_SEED = 777`
- **Use case**: **synthetic control** to verify coherent degradation behavior  
  (Dataset B is used only under `S1`)

---

## Experimental Design

### Split Policies (S1, S2)

- **S1 (stratified)**: 5-fold `StratifiedKFold` with shuffling and fixed seed
- **S2 (group-based)**: 5-fold `GroupKFold` enforcing **no patient overlap** via `subject_id`
- **Inner CV** for tuning is always **3 folds** (`INNER_FOLDS = 3`)
  - and is confined to the **outer-train** partition only

### Leakage Protocols (P0–P3)

Each protocol defines **where preprocessing occurs** relative to CV:

#### P0 — Strictly Controlled (No Global Leakage)

- All preprocessing is fold-safe:
  - imputation, scaling, encoding happen **inside folds**
- Nested CV for tuning uses only **outer-train**
- **Calibration split happens before tuning**:
  - split outer-train into `train_sub` and `cal_sub` (25% calibration)
  - tune on `train_sub` only
  - refit best model on `train_sub`
  - calibrate on `cal_sub` via a prefit calibrator

#### P1 — Global Imputation Leakage

- Global imputation applied to the **full dataset before splitting**
  - numeric: global median
  - categorical: global mode
- Fold pipelines omit imputers but keep fold-safe scaling/encoding
- Global imputation stats are saved as leakage artifacts

#### P2 — Global Scaling Leakage Only

- Global scaling applied to numeric columns using `nanmean/nanstd` computed on full data
- NaNs are preserved and imputed later within folds
- Fold pipelines include imputers but **skip scaling**
- Global means/scales saved as leakage artifacts

#### P3 — Global Transform + Selection (Broader Leakage)

- Fit a **global preprocessor** *(impute + scale + one-hot)* on the full dataset
- Transform full data once and compute mutual information (MI)
- Select top-k features in this **global feature space**
  - `K_A = 25` for Dataset A
  - `K_B = 6` for Dataset B
- During CV, folds reuse the **same global transform** and apply the same feature indices
- This explicitly leaks:
  - global preprocessing
  - global feature selection
- Selected indices and feature names are stored as leakage artifacts

### Models and Hyperparameters

The runner evaluates **five fixed model families**:

- `lr_l2` — L2 Logistic Regression (`LogisticRegression`)
  - grid: `C ∈ {0.1, 1.0, 10.0}`
- `svm_linear_cal` — Linear SVM (`LinearSVC`) + probability calibration
  - grid: `C ∈ {0.1, 1.0, 10.0}`
- `rf` — Random Forest (`RandomForestClassifier`, 600 trees)
  - grid: `max_depth ∈ {None, 6, 12}`
- `xgb` — XGBoost (`XGBClassifier`, 300 trees)
  - grid: `max_depth ∈ {3, 4, 5}`, `learning_rate ∈ {0.03, 0.05}`
- `extratrees` — ExtraTrees (`ExtraTreesClassifier`, 600 trees)
  - grid: `max_depth ∈ {None, 6, 12}`

Tuning is performed via `GridSearchCV` using **AUROC** as the tuning objective.

### Calibration Strategy

For models requiring calibration (**logistic regression** and **linear SVM**):

- Outer-train is split into:
  - `train_sub` (75%)
  - `cal_sub` (25%)
- **Tuning and model selection use only `train_sub`**
- Best model is refit on `train_sub`
- Calibrate on `cal_sub` via a **prefit** calibrator
- Calibration method: **sigmoid (Platt scaling)** by default

### Metrics and Winner Rule

Per outer fold:

- **AUROC** (`roc_auc_score`)
- **Average Precision (AP)** (`average_precision_score`)
- **Brier score** (`brier_score_loss`)
- **ECE** (10-bin implementation)

Configuration-level means and standard deviations are computed across the 5 outer folds.

**Winner selection is fixed lexicographically:**

1. Maximize mean **AUROC**
2. Tie-break with mean **AP**
3. Final tie-break with **lower** mean **Brier**

### Winner Flip Percentage (Phase 2)

Winner flip percentage measures decision instability across seeds:
Flip% = (# seeds where winner differs from baseline) / (total seeds) × 100

Baseline winner is taken from **Phase 1**, protocol **P0**, under the same split policy.

---

## Phases

The runner executes **Phase 1 → Phase 3 → Phase 2** in that order.

### Phase 1 — Main Audit Matrix (Dataset A)

- Tag: `PHASE1_MAIN`
- Dataset: A
- Splits: `S1`, `S2`
- Protocols: `P0`, `P1`, `P2`, `P3`
- Models: all 5
- Seed: `SEED_PHASE1 = 2026`

Total:
- `2 splits × 4 protocols × 5 models = 40 configurations`
- Each configuration: 5 outer folds + nested tuning

For **P0**, out-of-fold (OOF) predictions are stored for audit.

### Phase 3 — Synthetic Control (Dataset B)

- Tag: `PHASE3_SYN_CONTROL`
- Dataset: B
- Split: `S1` only
- Worlds:
  - `B_CLEAN`
  - `B_SYN_MISS` (MCAR 15%)
- Protocols: `P0`, `P1`
- Models: all 5
- Seed: `SEED_PHASE3 = 2040`

Total:
- `2 worlds × 2 protocols × 5 models = 20 configurations`

### Phase 2 — Seed Reproducibility (Dataset A, P0 only)

- Tag: `PHASE2_REPRO`
- Dataset: A
- Splits: `S1`, `S2`
- Protocol: `P0` only
- Seeds: `1001..1020` (20 seeds)
- Models: all 5

Total:
- `2 splits × 20 seeds × 5 models = 200 configurations`

Used to compute winner flip % relative to Phase 1 baseline winner.

---

## Outputs

All outputs are written to `results/` (created automatically).

Key files:

- `results/metrics_all.csv`
  - one row per **outer fold** and configuration
  - includes: `phase, dataset, split, protocol, model, seed, fold, AUROC, AP, Brier, ECE, best_params (JSON), ...`
- `results/summary_by_config.csv`
  - aggregated mean/std per configuration
- `results/winner_by_config.csv`
  - winner model per `(phase, dataset, split, protocol, seed)`
- `results/winner_flip_summary.csv`
  - winner flip % summary for Phase 2 (RQ3)
- `results/winner_by_seed_phase2.csv`
  - per-seed winners (Phase 2)
- `results/oof_P0_S1_<model>.npz`, `results/oof_P0_S2_<model>.npz`
  - OOF predictions for every P0 model in Phase 1, per split
  - fields: `y_true, y_prob, fold_id`
- `results/oof_P0_S1.npz`, `results/oof_P0_S2.npz`
  - canonical OOF files (copied from Phase 1 winner model), per split
- `results/leakage_artifacts/*.json`
  - leakage metadata for P1/P2/P3 (and optionally P0)
  - includes global stats, feature selection indices/names, and explicit leakage declaration
- `results/config_log.jsonl`
  - one JSONL record per launched configuration

---

## How to Run

1) Place the required datasets in the working directory:

- `full_analytic_dataset_mortality_all_admissions.csv`
- `Synthetic_Dataset_1500_Patients_precise.csv`

2) Run:

```bash
1. pip install numpy pandas scikit-learn xgboost tqdm
2. python jcsse_audit_runner.py
```
---

## Design Notes / Guarantees

This repository is designed as an **executable specification** of the protocol-level audit. The runner enforces the following guarantees:

### 1) Locked Protocol (Paper-Matched)
- **Seeds, hyperparameter grids, phase ordering, and configuration space are fixed** to match the paper.
- The runner is intentionally **not** a general-purpose training framework; it is a **locked audit runner** intended for exact reproduction.

### 2) Explicit, Auditable Leakage Variants
- All leakage-inducing steps are **explicitly defined** (P1–P3), **never implicit**.
- Each leakage protocol writes **protocol-specific artifacts** (e.g., global statistics, selected features) to disk for auditability and traceability.

### 3) Fold-Safe Baseline (P0)
The strict protocol `P0` is designed to prevent data re-use leakage and preserve evaluation integrity:
- All preprocessing (imputation, scaling, encoding) occurs **inside folds** via fold-safe pipelines.
- The outer-train partition is split into `train_sub` and `cal_sub` **before hyperparameter tuning**.
- Hyperparameter tuning and model selection are performed **only on `train_sub`**.
- Calibration is applied **only after refit**, using `cal_sub` via a **prefit** calibration wrapper.

### 4) Reproducible Synthetic Missingness (Dataset B)
- The synthetic control includes an MCAR perturbation:
  - **15% missingness**
  - applied to a **fixed feature list**
  - using a **fixed RNG seed**
- This makes degradation behavior **deterministic and reproducible** across environments (subject to library/version differences).

### 5) Defensive Handling of Common Data Issues
The runner is hardened against practical data problems that can silently break evaluation:
- **Hidden whitespace** or inconsistent formatting in column names (normalized before processing).
- **Fully-missing** or **infinite-valued** columns (sanitized/dropped safely when necessary).
- **Mixed-type columns** (string-like or inconsistent numeric formats) handled defensively to prevent pipeline crashes and unintended type leakage.

---

## Citation

If you use this codebase, audit framework, or reproduce results from this repository, please cite the following work and acknowledge the associated GitHub repository.

### GitHub Repository

Nattakitti Piyavechvirat.  
**Protocol-Level Audit Runner.**  
GitHub Repository: https://github.com/NattakittiP/Protocol_Audit

