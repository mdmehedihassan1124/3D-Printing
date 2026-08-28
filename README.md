# An Interpretable Machine-Learning Framework with Leakage-Free Validation for Predicting the Mechanical Behaviour of FFF-Printed PLA+ Components


**Authors:** M. A. Shadab Siddiquia, Mehedi Hassan Maruf b†, Md. Sakib Hossainb†, Arafath Mohivc†, Mahlet Tsedalu Adanea†, Ousman Mohammed Yimama†, Md. Sanaul Rabbib, Fahad Alama,d,       Mokarram Hossaine


a-Department of Materials Science and Engineering, King Fahd University of Petroleum and Minerals (KFUPM), Dhahran, Saudi Arabia

b-Department of Mechanical Engineering, Chittagong University of Engineering and Technology, Chattogram 4349, Bangladesh

c-Department of Mechanical Engineering, Ahsanullah University of Science & Technology, Dhaka, Bangladesh.

d-Interdisciplinary Research Center for Biosystems and Machines, King Fahd University of Petroleum and Minerals (KFUPM), Dhahran, Saudi Arabia

e-Zienkiewicz Institute for Modelling, Data and AI, Faculty of Science and Engineering, Swansea University, Swansea, SA1 8EN, United Kingdom

†These authors contributed equally to this work.

---
![Image alt](image_url)(https://github.com/mdmehedihassan1124/3D-Printing/blob/d19a94e987f56af43615f578fe40f81cf91a6aa5/Graphical%20Abstract.png)
## Abstract

Fused Filament Fabrication (FFF) is a popular Additive Manufacturing technique, but due to the anisotropy of the printed parts, there is not yet a reliable connection between the process parameters and the mechanical properties. An interpretable machine-learning framework is proposed for predicting the mechanical behavior of PLA+ specimens fabricated with different infill patterns, infill densities, and print orientations. The full-factorial design consisted of five infill patterns, four infill densities, and three different orientations of the print direction, leading to 60 factor combinations that were each replicated three times for 180 printed specimens. The compression strength and impact resistance were experimentally characterized and physics-informed surrogate responses were included to estimate the tensile strength, Young's modulus and elongation at break. A leakage free GroupKFold(10) cross validation strategy was used within the replicate-level data set to prevent information leakage between the training and testing folds. A standard validation scheme was used to compare 10 machine-learning and deep-learning models with various statistical indicators. Under leakage-free validation, compression strength and impact resistance proved difficult to predict from macroscopic parameters (best CC = 0.257 and 0.179). The physics-informed surrogate responses were reproduced with high accuracy (CC up to 0.979), which reflects the smoothness of their construction rather than predictive skill. To quantify the directional dependence, a compression-based Anisotropy Index (AI) was introduced. SHAP analysis identified infill density as the dominant driver of the surrogate responses by construction, whereas print orientation was most influential for Compression Strength and infill pattern for Impact Resistance, and closed form polynomial equations allowed for quick engineering estimate.

---

## Research Objectives

1.	To develop an interpretable ML framework for predicting the mechanical behaviour of FFF-printed PLA+ components. 
2.	To evaluate the effects of infill pattern, infill density, and print orientation on multiple mechanical properties. 
3.	To develop a leakage-free GroupKFold(10) validation framework that prevents information leakage from replicated specimens. 
4.	To benchmark ML and DL models for predicting tensile strength, compression strength, Young’s modulus, elongation at break, and impact resistance. 
5.	To quantify anisotropic behaviour of FFF-printed PLA+ using an Anisotropy Index (AI). 
6.	To interpret model predictions using SHAP analysis and identify the most influential process parameters. 
7.	To develop closed-form polynomial equations for rapid engineering estimation of the mechanical properties.

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
