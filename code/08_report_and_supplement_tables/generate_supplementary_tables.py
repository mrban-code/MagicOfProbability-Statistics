from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "02_可复用数据"
RESULTS = BASE / "03_可复用结果"
OUT = BASE / "04_补充明细表"


def normal_two_sided_p(z: float) -> float:
    if not math.isfinite(z):
        return math.nan
    return math.erfc(abs(z) / math.sqrt(2.0))


def significance(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def fit_ols(df: pd.DataFrame, y_col: str, x_cols: list[str], label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_df = df[[y_col, *x_cols]].dropna().copy()
    y = model_df[y_col].to_numpy(dtype=float)
    x = model_df[x_cols].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    names = ["Intercept", *x_cols]

    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    fitted = x @ beta
    resid = y - fitted

    n = len(y)
    p = len(names)
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    mse = rss / (n - p)
    cov = mse * xtx_inv
    se = np.sqrt(np.diag(cov))
    z = beta / se
    p_values = [normal_two_sided_p(float(v)) for v in z]

    y_std = float(np.std(y, ddof=1))
    standardized = [math.nan]
    for col, b in zip(x_cols, beta[1:]):
        standardized.append(float(b) * float(np.std(model_df[col], ddof=1)) / y_std)

    coefficients = pd.DataFrame(
        {
            "model": label,
            "term": names,
            "coefficient": beta,
            "std_error": se,
            "z_statistic_normal_approx": z,
            "p_value_normal_approx": p_values,
            "significance": [significance(v) for v in p_values],
            "standardized_coefficient": standardized,
            "interpretation_scale": [
                "PM2.5 units"
                if name == "Intercept"
                else "PM2.5 units per 1-unit increase; standardized column gives 1-SD effect"
                for name in names
            ],
        }
    )

    r2 = 1.0 - rss / tss
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p)
    rmse = math.sqrt(rss / n)
    mae = float(np.mean(np.abs(resid)))
    aic = n * (math.log(2 * math.pi * rss / n) + 1) + 2 * p
    bic = n * (math.log(2 * math.pi * rss / n) + 1) + math.log(n) * p
    f_stat = ((tss - rss) / (p - 1)) / (rss / (n - p))
    metrics = pd.DataFrame(
        {
            "model": [label],
            "n": [n],
            "predictors_excluding_intercept": [p - 1],
            "R2": [r2],
            "Adjusted_R2": [adj_r2],
            "RMSE": [rmse],
            "MAE": [mae],
            "AIC": [aic],
            "BIC": [bic],
            "F_statistic": [f_stat],
            "RSS": [rss],
        }
    )
    return metrics, coefficients


def add_numeric_definitions(sensitivity: pd.DataFrame, cutpoints: pd.DataFrame) -> pd.DataFrame:
    name_map = {
        "Temperature": "Temperature",
        "Humidity": "Humidity",
        "Wind speed": "WindSpeed",
        "CO": "CO",
        "NO2": "NO2",
        "SO2": "SO2",
        "O3": "O3",
        "PM10": "PM10",
    }
    units = {
        "PM2.5": "ug/m3",
        "Temperature": "deg C",
        "Humidity": "%",
        "WindSpeed": "m/s",
        "CO": "mg/m3",
        "NO2": "ug/m3",
        "SO2": "ug/m3",
        "O3": "ug/m3",
        "PM10": "ug/m3",
    }
    cut_map = cutpoints.set_index("变量")[["q33", "q67"]].to_dict("index")

    definitions = []
    for _, row in sensitivity.iterrows():
        label = row["variable_label"]
        state = row["state"]
        source_name = name_map.get(label)
        if row["variable"] == "Precipitation_rain":
            definition = "Rain" if state == "Rain" else "No rain"
            unit = "rain/no rain"
        elif source_name in cut_map:
            q33 = float(cut_map[source_name]["q33"])
            q67 = float(cut_map[source_name]["q67"])
            unit = units.get(source_name, "")
            if state == "Low":
                definition = f"<= {q33:.2f} {unit}"
            elif state == "Medium":
                definition = f"> {q33:.2f} and <= {q67:.2f} {unit}"
            else:
                definition = f"> {q67:.2f} {unit}"
        else:
            definition = state
            unit = ""
        definitions.append((definition, unit))

    detail = sensitivity.copy()
    detail["concrete_definition"] = [d[0] for d in definitions]
    detail["unit"] = [d[1] for d in definitions]
    detail["BN_P_high_percent"] = detail["BN_P_high"] * 100
    detail["delta_percentage_points"] = detail["delta_P"] * 100
    detail["abs_delta_percentage_points"] = detail["abs_delta_P"] * 100
    cols = [
        "variable_label",
        "category",
        "state",
        "concrete_definition",
        "n_days",
        "empirical_P_high",
        "BN_P_high",
        "BN_P_high_percent",
        "delta_P",
        "delta_percentage_points",
        "abs_delta_percentage_points",
    ]
    return detail[cols]


def contiguous_ranges(df: pd.DataFrame, mask: pd.Series) -> str:
    active = df.loc[mask, ["x"]]
    if active.empty:
        return ""

    xs = active["x"].to_list()
    ranges = []
    start = prev = xs[0]
    step = np.median(np.diff(df["x"])) if len(df) > 1 else 0.0
    for val in xs[1:]:
        if step and val - prev > step * 1.5:
            ranges.append((start, prev))
            start = val
        prev = val
    ranges.append((start, prev))
    return "; ".join(f"{a:.2f}-{b:.2f}" if abs(a - b) > 1e-9 else f"{a:.2f}" for a, b in ranges)


def zero_crossings(term_df: pd.DataFrame) -> str:
    xs = term_df["x"].to_numpy(dtype=float)
    ys = term_df["partial_effect"].to_numpy(dtype=float)
    roots = []
    for i in range(1, len(xs)):
        y0, y1 = ys[i - 1], ys[i]
        if y0 == 0:
            roots.append(xs[i - 1])
        elif y0 * y1 < 0:
            roots.append(xs[i - 1] + (0 - y0) * (xs[i] - xs[i - 1]) / (y1 - y0))
    return "; ".join(f"{r:.2f}" for r in roots)


def summarize_gam(partial: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for term, term_df in partial.groupby("smooth_term", sort=False):
        term_df = term_df.sort_values("x").reset_index(drop=True)
        min_row = term_df.loc[term_df["partial_effect"].idxmin()]
        max_row = term_df.loc[term_df["partial_effect"].idxmax()]
        rows.append(
            {
                "smooth_term": term,
                "x_min": term_df["x"].min(),
                "x_max": term_df["x"].max(),
                "effect_min_at_x": min_row["x"],
                "effect_min": min_row["partial_effect"],
                "effect_max_at_x": max_row["x"],
                "effect_max": max_row["partial_effect"],
                "zero_crossings_partial_effect": zero_crossings(term_df),
                "significant_positive_x_ranges_95ci": contiguous_ranges(term_df, term_df["lower_95"] > 0),
                "significant_negative_x_ranges_95ci": contiguous_ranges(term_df, term_df["upper_95"] < 0),
            }
        )
    return pd.DataFrame(rows)


def state_observed_summary(df: pd.DataFrame, cutpoints: pd.DataFrame) -> pd.DataFrame:
    variable_names = ["Temperature", "Humidity", "WindSpeed", "CO", "NO2", "SO2", "O3", "PM10"]
    units = {
        "Temperature": "deg C",
        "Humidity": "%",
        "WindSpeed": "m/s",
        "CO": "mg/m3",
        "NO2": "ug/m3",
        "SO2": "ug/m3",
        "O3": "ug/m3",
        "PM10": "ug/m3",
    }
    rows = []
    cut_map = cutpoints.set_index("变量")[["q33", "q67"]].to_dict("index")
    for var in variable_names:
        q33 = float(cut_map[var]["q33"])
        q67 = float(cut_map[var]["q67"])
        bins = [
            ("Low", df[var] <= q33, f"<= {q33:.2f} {units[var]}"),
            ("Medium", (df[var] > q33) & (df[var] <= q67), f"> {q33:.2f} and <= {q67:.2f} {units[var]}"),
            ("High", df[var] > q67, f"> {q67:.2f} {units[var]}"),
        ]
        for state, mask, definition in bins:
            sample = df.loc[mask]
            rows.append(
                {
                    "variable": var,
                    "state": state,
                    "concrete_definition": definition,
                    "n_days": len(sample),
                    "mean_PM25": sample["PM2.5"].mean(),
                    "median_PM25": sample["PM2.5"].median(),
                    "P_high_PM25_empirical": sample["PM2.5_high"].mean(),
                    "mean_variable_value": sample[var].mean(),
                }
            )
    for state, mask, definition in [
        ("No rain", df["IsRain"] == 0, "Precipitation = 0"),
        ("Rain", df["IsRain"] == 1, "Precipitation > 0"),
    ]:
        sample = df.loc[mask]
        rows.append(
            {
                "variable": "Precipitation",
                "state": state,
                "concrete_definition": definition,
                "n_days": len(sample),
                "mean_PM25": sample["PM2.5"].mean(),
                "median_PM25": sample["PM2.5"].median(),
                "P_high_PM25_empirical": sample["PM2.5_high"].mean(),
                "mean_variable_value": sample["Precipitation"].mean(),
            }
        )
    return pd.DataFrame(rows)


def consensus_table(
    bn_ranking: pd.DataFrame,
    rf_importance: pd.DataFrame,
    full_ols: pd.DataFrame,
    gam_smooth: pd.DataFrame,
) -> pd.DataFrame:
    lasso = pd.DataFrame(
        [
            ("CO", 14.300),
            ("NO2", 7.946),
            ("Lag_PM2.5", 6.952),
            ("season_Spring", 2.455),
            ("O3", 1.734),
            ("WindSpeed", 1.112),
            ("Precipitation", -0.732),
            ("month", -1.454),
            ("season_Winter", -1.513),
            ("SO2", -2.058),
        ],
        columns=["variable", "lasso_standardized_coefficient_from_main_report",],
    )

    label_to_var = {
        "Wind speed": "WindSpeed",
        "Temperature": "Temperature",
        "Humidity": "Humidity",
        "Precipitation": "Precipitation",
        "CO": "CO",
        "NO2": "NO2",
        "SO2": "SO2",
        "O3": "O3",
        "PM10": "PM10",
    }
    bn = bn_ranking.copy()
    bn["variable"] = bn["variable_label"].map(label_to_var).fillna(bn["variable"])
    bn = bn[["variable", "rank", "probability_range", "lowest_BN_P_high", "highest_BN_P_high"]].rename(
        columns={"rank": "BN_rank"}
    )

    rf = rf_importance.rename(columns={"Feature": "variable", "Importance_percent": "RF_importance_percent"})[
        ["variable", "RF_importance_percent"]
    ]
    ols = full_ols.loc[full_ols["term"] != "Intercept", ["term", "coefficient", "p_value_normal_approx", "standardized_coefficient"]].rename(
        columns={
            "term": "variable",
            "coefficient": "OLS_full_coefficient",
            "p_value_normal_approx": "OLS_p_value_normal_approx",
            "standardized_coefficient": "OLS_standardized_coefficient",
        }
    )
    gam = gam_smooth.rename(columns={"smooth_term": "variable", "drop_explained_deviance": "GAM_drop_explained_deviance"})[
        ["variable", "GAM_drop_explained_deviance"]
    ]

    variables = sorted(set(bn["variable"]) | set(rf["variable"]) | set(ols["variable"]) | set(gam["variable"]) | set(lasso["variable"]))
    out = pd.DataFrame({"variable": variables})
    for table in [bn, rf, lasso, ols, gam]:
        out = out.merge(table, on="variable", how="left")

    out["evidence_count"] = out[
        [
            "BN_rank",
            "RF_importance_percent",
            "lasso_standardized_coefficient_from_main_report",
            "OLS_standardized_coefficient",
            "GAM_drop_explained_deviance",
        ]
    ].notna().sum(axis=1)
    out = out.sort_values(["evidence_count", "RF_importance_percent", "GAM_drop_explained_deviance"], ascending=[False, False, False])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA / "analysis_dataset.csv")
    reg = pd.read_csv(DATA / "step5_full_ols_dataset.csv")
    cutpoints = pd.read_csv(DATA / "step2_discretize_cutpoints.csv")
    sensitivity = pd.read_csv(RESULTS / "bn_sensitivity_by_state.csv")
    bn_ranking = pd.read_csv(RESULTS / "bn_sensitivity_ranking.csv")
    rf_importance = pd.read_csv(RESULTS / "rf_feature_importance.csv")
    gam_smooth_terms = pd.read_csv(RESULTS / "gam_smooth_terms.csv")
    gam_partial = pd.read_csv(RESULTS / "gam_partial_effects.csv")

    weather_cols = [
        "Temperature",
        "Humidity",
        "WindSpeed",
        "Precipitation",
        "Lag_PM2.5",
        "season_Spring",
        "season_Summer",
        "season_Autumn",
    ]
    full_cols = [
        "Temperature",
        "Humidity",
        "WindSpeed",
        "Precipitation",
        "CO",
        "NO2",
        "SO2",
        "O3",
        "Lag_PM2.5",
        "season_Spring",
        "season_Summer",
        "season_Autumn",
    ]

    full_metrics, full_coef = fit_ols(reg, "PM2.5", full_cols, "Full OLS: meteorology + gaseous pollutants + lagged PM2.5")
    weather_metrics, weather_coef = fit_ols(reg, "PM2.5", weather_cols, "Weather-only OLS: meteorology + lagged PM2.5")
    pd.concat([full_metrics, weather_metrics], ignore_index=True).to_csv(OUT / "step5_model_comparison_full_vs_weather_only.csv", index=False)
    full_coef.to_csv(OUT / "ols_full_model_coefficients.csv", index=False)
    weather_coef.to_csv(OUT / "ols_weather_only_coefficients.csv", index=False)

    add_numeric_definitions(sensitivity, cutpoints).to_csv(OUT / "bn_concrete_state_risk_detail.csv", index=False)
    state_observed_summary(df, cutpoints).to_csv(OUT / "observed_pm25_by_concrete_variable_state.csv", index=False)
    summarize_gam(gam_partial).to_csv(OUT / "gam_partial_effect_summary_by_term.csv", index=False)
    consensus_table(bn_ranking, rf_importance, full_coef, gam_smooth_terms).to_csv(OUT / "cross_method_variable_consensus.csv", index=False)

    key_rows = [
        ("full_ols_R2", float(full_metrics.loc[0, "R2"])),
        ("weather_only_R2", float(weather_metrics.loc[0, "R2"])),
        ("R2_gain_from_gaseous_pollutants", float(full_metrics.loc[0, "R2"] - weather_metrics.loc[0, "R2"])),
        ("full_ols_adjusted_R2", float(full_metrics.loc[0, "Adjusted_R2"])),
        ("weather_only_adjusted_R2", float(weather_metrics.loc[0, "Adjusted_R2"])),
    ]
    pd.DataFrame(key_rows, columns=["item", "value"]).to_csv(OUT / "key_unification_numbers.csv", index=False)

    print("Generated supplementary tables in", OUT)


if __name__ == "__main__":
    main()
