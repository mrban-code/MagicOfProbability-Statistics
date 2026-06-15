# Magic of Probability & Statistics — PM₂.₅ Multifactor Analysis in Beijing

A multi-stage statistical association analysis of daily PM₂.₅ concentrations in Beijing, comparing the relative strength of meteorological conditions, pollutant co-movement, and short-term persistence.

## Overview

This study examines 1,462 daily observations for Beijing and uses a comparative workflow to answer: **Which type of information is most strongly associated with daily PM₂.₅ — weather, pollutant co-movement, or short-term persistence?**

Key finding: meteorological conditions are associated with PM₂.₅, but their standalone association is limited (R² ≈ 0.21). Pollutant co-movement (CO, NO₂, PM₁₀) and lagged PM₂.₅ provide substantially stronger statistical signals (R² ≈ 0.71–0.77).

## Methods

The analysis proceeds through five connected stages, plus a forecasting extension:

| Stage | Method | Purpose |
|-------|--------|---------|
| 1 | Descriptive statistics & correlation screening | Establish empirical baselines |
| 2 | Bayesian network (Naive Bayes structure) | State-based high-pollution probability analysis |
| 3 | Random forest & LASSO | Nonlinear prediction and regularized variable screening |
| 4 | Grouped OLS & full OLS | Compare association strength of weather vs. pollutants vs. persistence |
| 5 | Generalized Additive Model (GAM) | Nonlinear smooth-term association assessment |
| Extension | Ridge, random forest, ensemble forecasting | Test whether strong in-sample associations translate to out-of-sample prediction |

## Repository Structure

```
.
├── PM25_Beijing_multifactor_analysis.tex   # Full LaTeX manuscript
├── data/
│   ├── step1_descriptive_statistics.csv     # Descriptive stats tables
│   ├── step2_bayesian_network.csv           # BN state-risk outputs
│   ├── step2_discretize_cutpoints.csv        # Discretization thresholds
│   ├── step3_4_random_forest_lasso.csv       # RF feature importance & LASSO coefficients
│   ├── step5_full_ols_dataset.csv            # Full OLS regression coefficients
│   ├── step5_model_comparison_full_vs_weather_only.csv  # Grouped model comparison
│   ├── step6_gam_dataset.csv                 # GAM smooth-term results
│   ├── gam_*.csv                             # GAM linear/smooth term details
│   ├── rf_*.csv                              # Random forest metrics & importance
│   ├── ols_*.csv                             # OLS coefficients (full & weather-only)
│   ├── daily_model_*.csv                     # Daily forecasting model results
│   ├── trend_*.csv                           # Multi-horizon trend metrics
│   ├── bn_*.csv                              # BN sensitivity & scenario tables
│   ├── supplement_*.csv                      # Supplementary analysis outputs
│   ├── cross_method_variable_consensus.csv   # Cross-method consensus table
│   ├── variable_manifest.csv                 # Variable dictionary
│   ├── key_unification_numbers.csv           # Key numbers referenced in the paper
│   ├── analysis_dataset.csv                  # Main analysis dataset
│   ├── prediction_feature_dataset.csv         # Forecasting feature dataset
│   └── thresholds_and_risk_table.xlsx        # Risk threshold summary
└── figures/
    ├── desc_summary_panels.png               # Time series, histogram, boxplots, heatmap
    ├── bn_summary_panels.png                 # BN structure, state probabilities, sensitivity
    ├── rf_summary_panels.jpg                 # RF importance, observed vs. predicted
    ├── lasso_summary_panels.png              # LASSO cross-validation & coefficient path
    ├── mlr_diagnostic_panels.png             # Full OLS diagnostics
    ├── gam_diagnostic_importance_panels_clean.png  # GAM smooth-term ranking
    ├── supplement_group_model_r2_comparison.png    # Grouped model R² comparison
    ├── supplement_pm25_correlation_ranking.png     # Pearson correlation ranking
    └── best_ensemble_observed_vs_predicted.png     # Ensemble forecast performance
```

## Building the PDF

Compile the LaTeX manuscript with:

```bash
pdflatex PM25_Beijing_multifactor_analysis.tex
# or for a complete build with references:
latexmk -pdf PM25_Beijing_multifactor_analysis.tex
```

Requirements: A LaTeX distribution (TeX Live / MiKTeX) with standard packages (`booktabs`, `amsmath`, `graphicx`, `tabularx`, `hyperref`, `microtype`, etc.).

## Authors

Ban Jingyang, Chen Anyi, Chen Lichong, Feng Shuo, Lian Yuehan, and Wu Ziqi — all authors contributed equally.

## License

This is an academic project. Data sources and citation information are documented in the manuscript.
