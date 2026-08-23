# Machine Learning and Deep Learning-Driven Multi-Property Prediction and NSGA-II Pareto Optimisation of FFF 3D Printing Process Parameters

**Authors:** M. A. Shadab Siddiqui ; Mehedi Hassan Maruf;· Mahlet Adane · Ousman Yimam

---

## Abstract

This study presents a first-of-its-kind comprehensive machine learning (ML) and deep learning (DL) framework for simultaneous **prediction** and **multi-objective Pareto optimisation** of five key mechanical properties of Fused Filament Fabrication (FFF)-printed PLA+ specimens: **Tensile Strength (TS, MPa)**, **Compression Strength (CS, MPa)**, **Young's Modulus (E, GPa)**, **Elongation at Break (EB, %)**, and **Impact Resistance (IS, kJ/m²)**. A full-factorial experimental matrix of **180 specimens** — 60 unique parameter combinations × 3 replicates — spanning **5 infill patterns** (Line, Grid, Honeycomb, Cubic, Rectangular), **4 infill densities** (20–80%), and **3 print orientations** (0°/45°/90°) was designed and characterised.

A **dual-stream data strategy** ensures rigorous evaluation: all 180 replicate rows feed a **GroupKFold(10)** outer cross-validation (60 groups = unique parameter combinations, ~54 test rows/fold) that eliminates within-combination data leakage entirely — a methodological advance over the random CV splits used in most prior 3D printing ML studies. Replicate means (60 rows) are reserved exclusively for SHAP interpretability, NSGA-II optimisation, and analytical OLS equation derivation.

**10 prediction models** are benchmarked end-to-end — two linear models (LR, Ridge), two kernel models (SVR, GPR), two tree-ensemble models (Random Forest, XGBoost), three neural networks (MLP, 1D-CNN, Residual MLP), and one meta-ensemble (Stacking) — using **7 evaluation metrics** (CC, MAE, MSE, RMSE, RRSE, MAPE, PRMSE) with full train-vs-test per-fold overfitting diagnostics. The best models achieve CC > 0.95 for Compression Strength and Young’s Modulus, CC > 0.89 for Elongation at Break, while Tensile Strength (CC = 0.23) and Impact Resistance (CC = 0.17) reflect genuine inter-replicate variability. Top models are interpreted via SHAP and leveraged by a 5-objective NSGA-II optimiser (200 population × 150 generations) to produce a non-dominated Pareto front ranked by TOPSIS with application-weighted criteria.

---

## Research Objectives

1. **Benchmark 10 ML/DL models** for simultaneous five-property mechanical prediction from three FFF process parameters under a leakage-free GroupKFold(10) cross-validation protocol, reporting 7 evaluation metrics with train/test overfitting diagnostics.

2. **Derive closed-form OLS polynomial equations** (degree-2) validated by Leave-One-Out CV (n = 60) — enabling direct process planning and property prediction without any predictive software or trained model.

3. **Quantify feature influence** via SHAP (SHapley Additive exPlanations) for all five mechanical targets, providing physically interpretable rankings of infill density, pattern geometry, and print orientation across model families.

4. **Optimise five competing objectives simultaneously** using NSGA-II (200 population × 150 generations) and rank Pareto-optimal solutions via TOPSIS with application-specific importance weights.

5. **Introduce two novel scalar performance indices**: the **Anisotropy Index (AI)** — quantifying directional mechanical sensitivity between 0° and 90° orientations — and the **Mechanical Performance Index (MPI)** — a SHAP-weighted composite enabling single-number ranking of all 60 print configurations.

---

## Novel Contributions

| # | Novel Contribution | Impact |
|:---:|---|---|
| 1 | **10-model focused benchmark** spanning 4 paradigms: linear (LR, Ridge), kernel (SVR, GPR), tree-ensemble (RF, XGBoost), neural networks (MLP, 1D-CNN, Residual MLP) + Stacking meta-ensemble | Three DL architectures compared (shallow ANN vs deep MLP vs 1D-CNN) with only 2 linear baselines — balanced neural network coverage for a Q1 materials paper |
| 2 | **GroupKFold(10) on 180 replicate rows** — groups = parameter combinations | Eliminates within-combination data leakage absent from prior random-split studies; provides unbiased metric estimates |
| 3 | **Closed-form OLS polynomial equations** for all 5 mechanical properties, LOO-CV validated | Practitioner-ready: predict TS/CS/E/EB/IS from pattern+density+orientation without software; reproducible to ASTM/ISO standards |
| 4 | **SHAP interpretability across all 5 targets** using model-appropriate explainers (Tree/Linear/Kernel) | First systematic causal map of process parameter → mechanical property; identifies density as dominant driver (ε² > 0.60) |
| 5 | **Anisotropy Index (AI)** | Novel scalar quantifying pattern-specific directional sensitivity; reveals Honeycomb as most isotropic configuration |
| 6 | **Mechanical Performance Index (MPI)** | SHAP-weighted composite enabling rapid single-number ranking of 60 print configurations for multi-property trade-off decisions |

---

## Dataset Overview

| Parameter | Levels | Values |
|---|---|---|
| Infill Pattern | 5 | Line · Grid · Honeycomb · Cubic · Rectangular |
| Infill Density | 4 | 20% · 40% · 60% · 80% |
| Print Orientation | 3 | 0° · 45° · 90° |
| **Total combinations** | **60** | **× 3 replicates = 180 specimens** |

**Target properties:** TS (MPa) · CS (MPa) · E (GPa) · EB (%) · IS (kJ/m²)
**Material:** PLA+ filament · FFF/FDM process

## Section 0 — Research Workflow

```
RAW DATA: 180 specimens (60 combinations × 3 replicates)  |  Excel → Tensile + Impact sheets
                        │
          ┌─────────────┴──────────────────────────────────────────────┐
          │ STREAM A — CV / Training (180 rows)                        │ STREAM B — Inference (60 rows)
          │ All replicates retained                                     │ Replicate means only
          │                                                             │
          │  GroupKFold(k=10, 60 groups)                               │  Scalers fit on 60-row data
          │  ┌─────────────────────┐                                   │  SHAP · NSGA-II · OLS
          │  │ Fold 1…10           │                                   │
          │  │  Train: ~162 rows   │                                   │
          │  │  Test:  ~18 rows    │                                   │
          │  │  (6 combos × 3 rep) │                                   │
          │  └─────────────────────┘                                   │
          └─────────────────────────────────────────────────────────────┘
                        │
           EDA + Kruskal-Wallis significance test (Section 3)
                        │
           FEATURE ENGINEERING  (Section 4)
           Label-enc(3) · OHE+scaled(7) · Poly-deg2+scaled(9)
           HPO: 5-fold GridSearch on 60-row aggregated data
                        │
           ┌────────────┼──────────────┐
    CLASSICAL(3)    KERNEL(2)     TREE+BOOST(2)   DL+STACK(2)
    LR·Ridge       SVR·GPR       RF·XGBoost      DeepMLP
    Poly-Ridge                                   Stacking
           └────────────┴──────────────┘
                        │
           GroupKFold(10) OOF → per-fold TRAIN+TEST table
           Metrics: CC · MAE · MSE · RMSE · MAPE · PRMSE
                        │
    ┌───────────────────┴──────────────────────────────────┐
    │ SHAP beeswarm+grouped bar  │  OLS polynomial equations │
    │ (60-row final models)       │  (all 5 targets)         │
    └───────────────────┬─────────────────────────────────-─┘
                        │
           Sensitivity Analysis (density response curves)
           NSGA-II (5-objective, 200 pop × 150 gen) + TOPSIS
           Novel: Anisotropy Index (AI) · MPI radar
```
