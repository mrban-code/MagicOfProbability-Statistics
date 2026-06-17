#!/usr/bin/env python3
"""Supplementary analyses added for the final PM2.5 TeX report.

This script reproduces two final-report support artifacts:
1. supplement_group_model_comparison.csv and supplement_group_model_r2_comparison.png
2. supplement_pm25_correlation_ranking.csv and supplement_pm25_correlation_ranking.png

It expects to be run from this file's folder or from anywhere inside the project tree.
The script reads ../../data/analysis_dataset.csv relative to the cleaned report folder.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPORT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPORT_ROOT / "data"
FIG_DIR = REPORT_ROOT / "figures"
INPUT_CSV = DATA_DIR / "analysis_dataset.csv"


def fit_ols(df: pd.DataFrame, y_col: str, x_cols: list[str], model_name: str) -> dict[str, float | int | str]:
    model_df = df[[y_col, *x_cols]].dropna().copy()
    y = model_df[y_col].to_numpy(dtype=float)
    x = model_df[x_cols].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residuals = y - fitted
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    n = len(y)
    p = len(x_cols)
    r2 = 1.0 - rss / tss
    adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    return {
        "model": model_name,
        "n": n,
        "predictors_excluding_intercept": p,
        "R2": r2,
        "Adjusted_R2": adjusted_r2,
        "RMSE": math.sqrt(rss / n),
        "MAE": float(np.mean(np.abs(residuals))),
    }


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_horizontal_bars(
    output_path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    xmin: float,
    xmax: float,
    xlabel: str,
    colors: list[str],
) -> None:
    width, height = 1500, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font_title = get_font(30)
    font_axis = get_font(20)
    font_label = get_font(18)
    font_small = get_font(16)

    left, right, top, bottom = 540, 1340, 110, 740
    draw.text((left, 35), title, fill="#111827", font=font_title)

    n = len(labels)
    gap = 18
    bar_h = (bottom - top - (n - 1) * gap) / n

    for tick in np.linspace(xmin, xmax, 6):
        x = left + (tick - xmin) / (xmax - xmin) * (right - left)
        draw.line((x, top - 15, x, bottom + 10), fill="#e5e7eb", width=2)
        tick_label = f"{tick:.1f}" if xmax <= 1.1 else f"{tick:.0f}"
        draw.text((x - 20, bottom + 22), tick_label, fill="#4b5563", font=font_small)

    if xmin < 0 < xmax:
        x0 = left + (0 - xmin) / (xmax - xmin) * (right - left)
        draw.line((x0, top - 15, x0, bottom + 10), fill="#111827", width=3)
    else:
        x0 = left

    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * (bar_h + gap)
        draw.text((40, y + bar_h / 2 - 10), label, fill="#111827", font=font_label)
        x_value = left + (value - xmin) / (xmax - xmin) * (right - left)
        x1, x2 = min(x0, x_value), max(x0, x_value)
        draw.rounded_rectangle((x1, y, x2, y + bar_h), radius=7, fill=colors[i])
        text = f"{value:.3f}" if xmin >= 0 else f"{value:.2f}"
        tx = x2 + 12 if value >= 0 else x1 - 70
        draw.text((tx, y + bar_h / 2 - 10), text, fill="#111827", font=font_label)

    draw.text((left + 220, 825), xlabel, fill="#374151", font=font_axis)
    image.save(output_path)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df = df[df["valid_regression"].fillna(0).astype(int) == 1].copy()

    y_col = "PM2.5"
    season = ["season_Spring", "season_Summer", "season_Autumn"]
    weather = ["Temperature", "Humidity", "WindSpeed", "Precipitation"]
    gases = ["CO", "NO2", "SO2", "O3"]
    lag = ["Lag_PM2.5"]
    particulate = ["PM10"]

    model_specs = [
        ("Weather only + season", weather + season),
        ("Weather + lag + season", weather + lag + season),
        ("Gaseous pollutants + season", gases + season),
        ("Gaseous pollutants + lag + season", gases + lag + season),
        ("Weather + gases + season", weather + gases + season),
        ("Full: weather + gases + lag + season", weather + gases + lag + season),
        ("Full plus PM10 diagnostic", weather + gases + particulate + lag + season),
    ]

    comparison = pd.DataFrame([fit_ols(df, y_col, cols, name) for name, cols in model_specs])
    comparison.to_csv(DATA_DIR / "supplement_group_model_comparison.csv", index=False)

    vars_for_corr = weather + gases + particulate + lag + ["AQI"]
    corr_rows = []
    for var in vars_for_corr:
        pair = df[[y_col, var]].dropna()
        pearson = pair[y_col].corr(pair[var], method="pearson")
        spearman = pair[y_col].rank().corr(pair[var].rank(), method="pearson")
        corr_rows.append(
            {
                "variable": var,
                "pearson_corr_with_PM2.5": pearson,
                "spearman_corr_with_PM2.5": spearman,
                "abs_pearson": abs(pearson),
                "n": len(pair),
            }
        )
    corr = pd.DataFrame(corr_rows).sort_values("abs_pearson", ascending=False)
    corr.to_csv(DATA_DIR / "supplement_pm25_correlation_ranking.csv", index=False)

    plot_comp = comparison[comparison["model"] != "Full plus PM10 diagnostic"].copy().iloc[::-1]
    draw_horizontal_bars(
        FIG_DIR / "supplement_group_model_r2_comparison.png",
        "Group-wise OLS comparison for PM2.5",
        list(plot_comp["model"]),
        list(plot_comp["R2"]),
        0.0,
        0.85,
        "In-sample R-squared",
        ["#0f766e", "#0891b2", "#1d4ed8", "#2563eb", "#9ca3af", "#6b7280"],
    )

    plot_corr = corr.head(10).iloc[::-1]
    draw_horizontal_bars(
        FIG_DIR / "supplement_pm25_correlation_ranking.png",
        "Correlation ranking of candidate variables",
        list(plot_corr["variable"]),
        list(plot_corr["pearson_corr_with_PM2.5"]),
        -0.55,
        1.0,
        "Pearson correlation with PM2.5",
        ["#dc2626" if value >= 0 else "#2563eb" for value in plot_corr["pearson_corr_with_PM2.5"]],
    )

    print(f"Wrote {DATA_DIR / 'supplement_group_model_comparison.csv'}")
    print(f"Wrote {DATA_DIR / 'supplement_pm25_correlation_ranking.csv'}")
    print(f"Wrote {FIG_DIR / 'supplement_group_model_r2_comparison.png'}")
    print(f"Wrote {FIG_DIR / 'supplement_pm25_correlation_ranking.png'}")


if __name__ == "__main__":
    main()
