# Multifactor Association Analysis of Beijing PM₂.₅

**Meteorological Conditions, Pollutant Co-movement, and Short-term Persistence**

## Authors

Ban Jinyang, Chen Anyi, Chen Lichong, Feng Shuo, Lian Yuehan, and Wu Ziqi

*Probability Theory and Mathematical Statistics (B) — Course Project, June 2026*

## Overview

This project investigates the multifactor associations driving Beijing's daily PM₂.₅ concentrations. Rather than treating PM₂.₅ as a single-factor problem, we systematically compare three families of predictors — meteorological conditions, gaseous co-pollutants, and short-term temporal persistence — using a diverse suite of statistical and machine learning methods.

## Methods

The analysis proceeds through the following steps:

| Step | Method | Purpose |
|------|--------|---------|
| 1 | Descriptive Statistics | Summarize variable distributions, correlations, and seasonal patterns |
| 2 | Bayesian Network | Learn probabilistic dependency structure and quantify conditional risk across discretized states |
| 3–4 | Random Forest & LASSO | Identify key predictors via tree-based importance and L1-regularized selection |
| 5 | OLS Linear Regression | Full model and weather-only model comparison to isolate marginal contributions |
| 6 | Generalized Additive Model (GAM) | Capture nonlinear marginal effects of each predictor |
| 7 | Forecasting (Ensemble) | Short-term PM₂.₅ prediction using Random Forest, XGBoost, and ensemble blending |
| 8 | Report & Supplementary Tables | Generate final manuscript tables and supplementary materials |

## Directory Structure

```
.
├── PM25_Beijing_multifactor_analysis.tex   # LaTeX manuscript
├── README.md
├── code/
│   ├── 00_final_supplement/                # Supplementary analysis for final manuscript
│   ├── 02_bayesian_network/                # Bayesian network structure learning & inference
│   ├── 05_regression/                      # OLS regression (full & weather-only models)
│   ├── 06_gam/                             # Generalized Additive Models
│   ├── 07_forecasting/                     # ML-based PM₂.₅ prediction
│   └── 08_report_and_supplement_tables/    # Report generation & supplementary tables
├── data/                                   # 33 CSV/XLSX datasets (input & output)
└── figures/                                # 9 figures (PNG/JPG)
```

## Key Findings

- Gaseous co-pollutants (particularly CO and NO₂) and short-term persistence (lag-1 PM₂.₅) are the dominant predictors, with meteorological variables playing a secondary but complementary role.
- Bayesian network analysis reveals concrete risk scenarios: high CO combined with low wind speed yields sharply elevated PM₂.₅ probabilities.
- Ensemble forecasting achieves strong predictive performance, with the blended model outperforming individual learners.
- Cross-method consensus analysis identifies a core set of variables that are consistently important across all modeling approaches.
