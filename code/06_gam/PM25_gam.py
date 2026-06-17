#!/usr/bin/env python3
"""Step 6: penalized-spline GAM analysis for daily PM2.5."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_CSV = PROJECT_DIR / "变量" / "step6_gam.csv"

TARGET = "PM2.5"
SMOOTH_TERMS = ("WindSpeed", "Humidity", "NO2", "CO", "O3")
LINEAR_TERMS = ("Precipitation", "Temperature", "Lag_PM2.5", "month")
N_BASIS = 9
DEGREE = 3
LAMBDA_GRID = np.logspace(-3, 5, 33)

COLORS = {
    "background": "#FFFFFF",
    "panel": "#FAFAFA",
    "grid": "#C8C8C8",
    "text": "#262626",
    "muted": "#666666",
    "navy": "#262626",
    "blue": "#3498DB",
    "teal": "#3498DB",
    "orange": "#3498DB",
    "coral": "#FF0000",
    "purple": "#3498DB",
    "confidence": "#C9E6F6",
}


@dataclass
class SmoothSpec:
    name: str
    knots: np.ndarray
    center: np.ndarray
    column_slice: slice


@dataclass
class FitResult:
    beta: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    rss: float
    edf: float
    gcv: float
    covariance: np.ndarray
    sigma2: float
    aic: float
    r2: float
    adjusted_r2: float
    rmse: float
    mae: float


def read_data(path: Path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {TARGET, *SMOOTH_TERMS, *LINEAR_TERMS, "date"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    dates = [row["date"] for row in rows]
    data = {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in (TARGET, *SMOOTH_TERMS, *LINEAR_TERMS)
    }
    if any(np.isnan(values).any() for values in data.values()):
        raise ValueError("GAM input contains missing numeric values.")
    return dates, data


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_knots(x: np.ndarray, n_basis: int = N_BASIS, degree: int = DEGREE) -> np.ndarray:
    interior_count = n_basis - degree - 1
    interior = np.quantile(x, np.linspace(0, 1, interior_count + 2)[1:-1])
    if np.unique(interior).size != interior_count:
        interior = np.linspace(float(x.min()), float(x.max()), interior_count + 2)[1:-1]
    return np.concatenate(
        (
            np.repeat(float(x.min()), degree + 1),
            interior,
            np.repeat(float(x.max()), degree + 1),
        )
    )


def bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int = DEGREE) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    basis = np.zeros((x.size, len(knots) - 1), dtype=float)
    for index in range(len(knots) - 1):
        basis[:, index] = ((x >= knots[index]) & (x < knots[index + 1])).astype(float)
    for current_degree in range(1, degree + 1):
        next_basis = np.zeros((x.size, len(knots) - current_degree - 1), dtype=float)
        for index in range(next_basis.shape[1]):
            left_denom = knots[index + current_degree] - knots[index]
            right_denom = knots[index + current_degree + 1] - knots[index + 1]
            if left_denom > 0:
                next_basis[:, index] += (x - knots[index]) / left_denom * basis[:, index]
            if right_denom > 0:
                next_basis[:, index] += (
                    (knots[index + current_degree + 1] - x) / right_denom * basis[:, index + 1]
                )
        basis = next_basis
    at_right_edge = np.isclose(x, knots[-1])
    basis[at_right_edge, :] = 0.0
    basis[at_right_edge, -1] = 1.0
    return basis


def second_difference_penalty(size: int) -> np.ndarray:
    difference = np.diff(np.eye(size), n=2, axis=0)
    return difference.T @ difference


def build_design(data: Mapping[str, np.ndarray]):
    n = len(data[TARGET])
    columns = [np.ones((n, 1), dtype=float)]
    column_names = ["Intercept"]
    linear_scaling: Dict[str, Tuple[float, float]] = {}
    for name in LINEAR_TERMS:
        mean = float(data[name].mean())
        std = float(data[name].std(ddof=1))
        linear_scaling[name] = (mean, std)
        columns.append(((data[name] - mean) / std).reshape(-1, 1))
        column_names.append(name)

    penalty_blocks = [np.zeros((1 + len(LINEAR_TERMS), 1 + len(LINEAR_TERMS)), dtype=float)]
    smooth_specs: Dict[str, SmoothSpec] = {}
    start = 1 + len(LINEAR_TERMS)
    for name in SMOOTH_TERMS:
        knots = make_knots(data[name])
        basis = bspline_basis(data[name], knots)
        center = basis.mean(axis=0)
        centered_basis = basis - center
        columns.append(centered_basis)
        smooth_specs[name] = SmoothSpec(name=name, knots=knots, center=center, column_slice=slice(start, start + N_BASIS))
        start += N_BASIS
        penalty_blocks.append(second_difference_penalty(N_BASIS))
        column_names.extend(f"{name}_spline_{index + 1}" for index in range(N_BASIS))

    design = np.column_stack(columns)
    penalty = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    cursor = 1 + len(LINEAR_TERMS)
    for block in penalty_blocks[1:]:
        penalty[cursor : cursor + N_BASIS, cursor : cursor + N_BASIS] = block
        cursor += N_BASIS
    return design, penalty, linear_scaling, smooth_specs, column_names


def fit_penalized(design: np.ndarray, y: np.ndarray, penalty: np.ndarray, smoothing_lambda: float) -> FitResult:
    n = len(y)
    xtx = design.T @ design
    ridge = np.eye(design.shape[1]) * 1e-9
    ridge[0, 0] = 0.0
    system = xtx + smoothing_lambda * penalty + ridge
    system_inverse = np.linalg.pinv(system, rcond=1e-11)
    beta = system_inverse @ design.T @ y
    fitted = design @ beta
    residuals = y - fitted
    rss = float(residuals @ residuals)
    edf = float(np.trace(system_inverse @ xtx))
    sigma2 = rss / max(n - edf, 1.0)
    covariance = sigma2 * system_inverse @ xtx @ system_inverse
    gcv = n * rss / max((n - edf) ** 2, 1e-12)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss
    adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1.0) / max(n - edf, 1.0)
    aic = n * (math.log(2.0 * math.pi) + 1.0 + math.log(rss / n)) + 2.0 * edf
    return FitResult(
        beta=beta,
        fitted=fitted,
        residuals=residuals,
        rss=rss,
        edf=edf,
        gcv=gcv,
        covariance=covariance,
        sigma2=sigma2,
        aic=aic,
        r2=r2,
        adjusted_r2=adjusted_r2,
        rmse=float(np.sqrt(np.mean(residuals**2))),
        mae=float(np.mean(np.abs(residuals))),
    )


def select_lambda(design: np.ndarray, y: np.ndarray, penalty: np.ndarray):
    search_rows = []
    best_lambda = None
    best_fit = None
    for smoothing_lambda in LAMBDA_GRID:
        fit = fit_penalized(design, y, penalty, float(smoothing_lambda))
        search_rows.append(
            {
                "lambda": round(float(smoothing_lambda), 10),
                "edf": round(fit.edf, 8),
                "GCV": round(fit.gcv, 8),
                "AIC": round(fit.aic, 8),
                "RMSE": round(fit.rmse, 8),
            }
        )
        if best_fit is None or fit.gcv < best_fit.gcv:
            best_lambda = float(smoothing_lambda)
            best_fit = fit
    return best_lambda, best_fit, search_rows


def fit_linear_baseline(data: Mapping[str, np.ndarray]) -> FitResult:
    names = (*SMOOTH_TERMS, *LINEAR_TERMS)
    columns = [np.ones((len(data[TARGET]), 1), dtype=float)]
    for name in names:
        values = data[name]
        columns.append(((values - values.mean()) / values.std(ddof=1)).reshape(-1, 1))
    design = np.column_stack(columns)
    penalty = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    return fit_penalized(design, data[TARGET], penalty, 0.0)


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 250
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * beta_continued_fraction(b, a, 1.0 - x) / b


def f_survival(value: float, numerator_df: float, denominator_df: float) -> float:
    if value <= 0:
        return 1.0
    x = denominator_df / (denominator_df + numerator_df * value)
    return regularized_incomplete_beta(denominator_df / 2.0, numerator_df / 2.0, x)


def significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def format_p_value(p_value: float) -> str:
    if p_value < 1e-10:
        return "<1e-10"
    return f"{p_value:.3g}"


def smooth_term_tests(
    design: np.ndarray,
    y: np.ndarray,
    penalty: np.ndarray,
    fit: FitResult,
    smoothing_lambda: float,
    specs: Mapping[str, SmoothSpec],
) -> List[Dict[str, object]]:
    rows = []
    tss = float(((y - y.mean()) ** 2).sum())
    for name in SMOOTH_TERMS:
        term_slice = specs[name].column_slice
        keep = np.ones(design.shape[1], dtype=bool)
        keep[term_slice] = False
        reduced_design = design[:, keep]
        reduced_penalty = penalty[np.ix_(keep, keep)]
        reduced_fit = fit_penalized(reduced_design, y, reduced_penalty, smoothing_lambda)
        df_difference = max(fit.edf - reduced_fit.edf, 1e-6)
        rss_difference = max(reduced_fit.rss - fit.rss, 0.0)
        f_statistic = (rss_difference / df_difference) / fit.sigma2
        p_value = f_survival(f_statistic, df_difference, len(y) - fit.edf)
        rows.append(
            {
                "smooth_term": name,
                "effective_df": round(df_difference, 6),
                "F_statistic": round(f_statistic, 6),
                "p_value": max(p_value, 1e-300),
                "significance": significance_label(p_value),
                "drop_RSS": round(rss_difference, 6),
                "drop_explained_deviance": round(rss_difference / tss, 8),
            }
        )
    rows.sort(key=lambda row: float(row["drop_explained_deviance"]), reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def linear_term_rows(fit: FitResult, data: Mapping[str, np.ndarray]) -> List[Dict[str, object]]:
    rows = []
    for index, name in enumerate(("Intercept", *LINEAR_TERMS)):
        coefficient = float(fit.beta[index])
        std_error = math.sqrt(max(float(fit.covariance[index, index]), 0.0))
        statistic = coefficient / std_error if std_error else float("inf")
        p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
        rows.append(
            {
                "term": name,
                "coefficient": round(coefficient, 8),
                "std_error": round(std_error, 8),
                "z_statistic_approx": round(statistic, 6),
                "p_value_approx": round(p_value, 10),
                "significance": significance_label(p_value),
                "coefficient_scale": "PM2.5 units" if name == "Intercept" else "PM2.5 units per 1 SD increase",
            }
        )
    return rows


def partial_effect_rows(
    fit: FitResult,
    data: Mapping[str, np.ndarray],
    specs: Mapping[str, SmoothSpec],
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, np.ndarray]]]:
    rows = []
    curves: Dict[str, Dict[str, np.ndarray]] = {}
    for name in SMOOTH_TERMS:
        spec = specs[name]
        low, high = np.quantile(data[name], [0.01, 0.99])
        grid = np.linspace(float(low), float(high), 120)
        basis = bspline_basis(grid, spec.knots) - spec.center
        coefficients = fit.beta[spec.column_slice]
        covariance = fit.covariance[spec.column_slice, spec.column_slice]
        effect = basis @ coefficients
        variance = np.einsum("ij,jk,ik->i", basis, covariance, basis)
        std_error = np.sqrt(np.maximum(variance, 0.0))
        lower = effect - 1.96 * std_error
        upper = effect + 1.96 * std_error
        curves[name] = {"x": grid, "effect": effect, "lower": lower, "upper": upper}
        for x_value, center, lower_value, upper_value in zip(grid, effect, lower, upper):
            rows.append(
                {
                    "smooth_term": name,
                    "x": round(float(x_value), 8),
                    "partial_effect": round(float(center), 8),
                    "lower_95": round(float(lower_value), 8),
                    "upper_95": round(float(upper_value), 8),
                }
            )
    return rows, curves


def find_font(size: int, bold: bool = False):
    names = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def canvas(title: str, subtitle: str, width: int = 1800, height: int = 1000):
    image = Image.new("RGB", (width, height), COLORS["background"])
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 28), title, fill=COLORS["text"], font=find_font(36), anchor="ma")
    return image, draw


def scale(value: float, domain: Tuple[float, float], target: Tuple[int, int]) -> int:
    low, high = domain
    if math.isclose(low, high):
        return (target[0] + target[1]) // 2
    return round(target[0] + (value - low) / (high - low) * (target[1] - target[0]))


def nice_range(values: np.ndarray, include_zero: bool = False) -> Tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    padding = max((high - low) * 0.12, 1e-6)
    return low - padding, high + padding


def draw_panel_axes(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    x_domain: Tuple[float, float],
    y_domain: Tuple[float, float],
    x_label: str,
    title: str,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    left, top, right, bottom = box
    draw.rectangle(box, fill=COLORS["panel"], outline=COLORS["grid"], width=2)
    plot_x = (left + 68, right - 18)
    plot_y = (bottom - 56, top + 55)
    draw.text(((left + right) // 2, top + 14), title, fill=COLORS["text"], font=find_font(20), anchor="ma")
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x_value = x_domain[0] + fraction * (x_domain[1] - x_domain[0])
        x = scale(x_value, x_domain, plot_x)
        draw.line((x, plot_y[0], x, plot_y[1]), fill=COLORS["grid"], width=2)
        draw.text((x, plot_y[0] + 11), f"{x_value:.2g}", fill=COLORS["muted"], font=find_font(15), anchor="ma")
        y_value = y_domain[0] + fraction * (y_domain[1] - y_domain[0])
        y = scale(y_value, y_domain, plot_y)
        draw.line((plot_x[0], y, plot_x[1], y), fill=COLORS["grid"], width=2)
        draw.text((plot_x[0] - 10, y), f"{y_value:.1f}", fill=COLORS["muted"], font=find_font(14), anchor="rm")
    draw.rectangle((plot_x[0], plot_y[1], plot_x[1], plot_y[0]), outline=COLORS["grid"], width=2)
    draw.text(((plot_x[0] + plot_x[1]) // 2, bottom - 18), x_label, fill=COLORS["muted"], font=find_font(16), anchor="mm")
    return plot_x, plot_y


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    fill: str,
    width: int = 4,
    dash: int = 18,
    gap: int = 12,
) -> None:
    x1, y1 = start
    x2, y2 = end
    distance = math.hypot(x2 - x1, y2 - y1)
    if distance == 0:
        return
    unit_x = (x2 - x1) / distance
    unit_y = (y2 - y1) / distance
    cursor = 0.0
    while cursor < distance:
        segment_end = min(cursor + dash, distance)
        draw.line(
            (
                round(x1 + unit_x * cursor),
                round(y1 + unit_y * cursor),
                round(x1 + unit_x * segment_end),
                round(y1 + unit_y * segment_end),
            ),
            fill=fill,
            width=width,
        )
        cursor += dash + gap


def draw_vertical_label(image: Image.Image, text: str, center: Tuple[int, int], font_size: int = 23) -> None:
    font = find_font(font_size)
    scratch = Image.new("RGBA", (600, 80), (255, 255, 255, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    scratch_draw.text((300, 40), text, fill=COLORS["text"], font=font, anchor="mm")
    cropped = scratch.crop(scratch.getbbox())
    rotated = cropped.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(rotated, (center[0] - rotated.width // 2, center[1] - rotated.height // 2), rotated)


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_domain: Tuple[float, float],
    y_domain: Tuple[float, float],
    plot_x: Tuple[int, int],
    plot_y: Tuple[int, int],
    color: str,
    width: int = 3,
) -> None:
    points = [
        (scale(float(x), x_domain, plot_x), scale(float(y), y_domain, plot_y))
        for x, y in zip(x_values, y_values)
    ]
    draw.line(points, fill=color, width=width, joint="curve")


def figure_smooth_curves(curves: Mapping[str, Mapping[str, np.ndarray]], tests: Sequence[Mapping[str, object]]) -> None:
    image, draw = canvas(
        "Fig. 8 GAM Partial Effects for Daily PM2.5 (Step 6)",
        "",
    )
    test_lookup = {row["smooth_term"]: row for row in tests}
    boxes = (
        (55, 100, 585, 505),
        (635, 100, 1165, 505),
        (1215, 100, 1745, 505),
        (345, 555, 875, 960),
        (925, 555, 1455, 960),
    )
    for name, box in zip(SMOOTH_TERMS, boxes):
        curve = curves[name]
        x_domain = (float(curve["x"].min()), float(curve["x"].max()))
        y_domain = nice_range(np.concatenate((curve["lower"], curve["upper"])), include_zero=True)
        test = test_lookup[name]
        title = f"{name}   EDF={float(test['effective_df']):.2f}  p={format_p_value(float(test['p_value']))} {test['significance']}"
        plot_x, plot_y = draw_panel_axes(draw, box, x_domain, y_domain, name, title)
        zero_y = scale(0.0, y_domain, plot_y)
        draw.line((plot_x[0], zero_y, plot_x[1], zero_y), fill=COLORS["grid"], width=2)
        polygon = [
            (scale(float(x), x_domain, plot_x), scale(float(y), y_domain, plot_y))
            for x, y in zip(curve["x"], curve["lower"])
        ]
        polygon += [
            (scale(float(x), x_domain, plot_x), scale(float(y), y_domain, plot_y))
            for x, y in zip(curve["x"][::-1], curve["upper"][::-1])
        ]
        draw.polygon(polygon, fill=COLORS["confidence"])
        draw_polyline(draw, curve["x"], curve["effect"], x_domain, y_domain, plot_x, plot_y, COLORS["blue"], width=4)
    save_image(image, "fig1_gam_smooth_effects.png")


def figure_observed_vs_fitted(y: np.ndarray, fitted: np.ndarray, metrics: Mapping[str, float]) -> None:
    image, draw = canvas(
        "Fig. 9 Observed vs Fitted PM2.5 (Step 6)",
        "",
    )
    left, top, right, bottom = 180, 95, 1680, 885
    draw.rectangle((left, top, right, bottom), fill=COLORS["panel"], outline=COLORS["grid"], width=2)
    low = min(float(y.min()), float(fitted.min()), 0.0) - 5.0
    high = max(float(y.max()), float(fitted.max())) * 1.05
    domain = (low, high)
    for tick in np.linspace(low, high, 8):
        x = scale(float(tick), domain, (left, right))
        yy = scale(float(tick), domain, (bottom, top))
        draw.line((x, bottom, x, top), fill=COLORS["grid"], width=2)
        draw.line((left, yy, right, yy), fill=COLORS["grid"], width=2)
        draw.text((x, bottom + 15), f"{tick:.0f}", fill=COLORS["text"], font=find_font(19), anchor="ma")
        draw.text((left - 14, yy), f"{tick:.0f}", fill=COLORS["text"], font=find_font(19), anchor="rm")
    draw_dashed_line(draw, (left, bottom), (right, top), fill=COLORS["coral"], width=5)
    for actual, estimate in zip(y, fitted):
        x = scale(float(actual), domain, (left, right))
        yy = scale(float(estimate), domain, (bottom, top))
        draw.ellipse((x - 6, yy - 6, x + 6, yy + 6), fill=COLORS["blue"], outline="#2C7FB8", width=1)
    draw.text(((left + right) // 2, 955), "Observed PM2.5 (ug/m3)", fill=COLORS["text"], font=find_font(23), anchor="mm")
    draw_vertical_label(image, "Fitted PM2.5 (ug/m3)", (65, (top + bottom) // 2))
    draw.rectangle((210, 120, 430, 170), fill=COLORS["background"], outline=COLORS["grid"], width=2)
    draw_dashed_line(draw, (230, 145), (290, 145), fill=COLORS["coral"], width=5, dash=15, gap=8)
    draw.text((310, 132), "1:1 Line", fill=COLORS["text"], font=find_font(20))
    save_image(image, "fig2_observed_vs_fitted.png")


def figure_residuals(fitted: np.ndarray, residuals: np.ndarray) -> None:
    image, draw = canvas(
        "Fig. 10 Residuals vs Fitted Values (Step 6)",
        "",
    )
    left, top, right, bottom = 180, 95, 1680, 885
    draw.rectangle((left, top, right, bottom), fill=COLORS["panel"], outline=COLORS["grid"], width=2)
    x_domain = nice_range(fitted)
    y_domain = nice_range(residuals, include_zero=True)
    for tick in np.linspace(x_domain[0], x_domain[1], 8):
        x = scale(float(tick), x_domain, (left, right))
        draw.line((x, bottom, x, top), fill=COLORS["grid"], width=2)
        draw.text((x, bottom + 15), f"{tick:.0f}", fill=COLORS["text"], font=find_font(19), anchor="ma")
    for tick in np.linspace(y_domain[0], y_domain[1], 8):
        y = scale(float(tick), y_domain, (bottom, top))
        draw.line((left, y, right, y), fill=COLORS["grid"], width=2)
        draw.text((left - 14, y), f"{tick:.0f}", fill=COLORS["text"], font=find_font(19), anchor="rm")
    zero_y = scale(0.0, y_domain, (bottom, top))
    draw_dashed_line(draw, (left, zero_y), (right, zero_y), fill=COLORS["coral"], width=5)
    for estimate, residual in zip(fitted, residuals):
        x = scale(float(estimate), x_domain, (left, right))
        y = scale(float(residual), y_domain, (bottom, top))
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=COLORS["blue"], outline="#2C7FB8", width=1)
    draw.text(((left + right) // 2, 955), "Fitted PM2.5 (ug/m3)", fill=COLORS["text"], font=find_font(23), anchor="mm")
    draw_vertical_label(image, "Residuals", (68, (top + bottom) // 2))
    save_image(image, "fig3_residuals_vs_fitted.png")


def figure_residual_distribution(residuals: np.ndarray) -> None:
    image, draw = canvas(
        "Fig. 11 Distribution of GAM Residuals (Step 6)",
        "",
    )
    left, top, right, bottom = 170, 95, 1690, 885
    draw.rectangle((left, top, right, bottom), fill=COLORS["panel"], outline=COLORS["grid"], width=2)
    counts, edges = np.histogram(residuals, bins=40)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_width = float(edges[1] - edges[0])
    bandwidth = max(1.06 * float(residuals.std(ddof=1)) * len(residuals) ** (-0.2), bin_width * 0.7)
    curve_x = np.linspace(float(edges[0]), float(edges[-1]), 320)
    standardized = (curve_x[:, None] - residuals[None, :]) / bandwidth
    density = np.exp(-0.5 * standardized**2).sum(axis=1) / (len(residuals) * bandwidth * math.sqrt(2.0 * math.pi))
    curve_y = density * len(residuals) * bin_width
    x_domain = nice_range(np.asarray([edges[0], edges[-1]]))
    y_domain = (0.0, max(float(counts.max()), float(curve_y.max())) * 1.06)
    for tick in np.linspace(x_domain[0], x_domain[1], 8):
        x = scale(float(tick), x_domain, (left, right))
        draw.line((x, bottom, x, top), fill=COLORS["grid"], width=2)
        draw.text((x, bottom + 15), f"{tick:.0f}", fill=COLORS["text"], font=find_font(19), anchor="ma")
    for tick in np.linspace(0.0, y_domain[1], 7):
        y = scale(float(tick), y_domain, (bottom, top))
        draw.line((left, y, right, y), fill=COLORS["grid"], width=2)
        draw.text((left - 14, y), f"{tick:.0f}", fill=COLORS["text"], font=find_font(19), anchor="rm")
    for count, edge_left, edge_right in zip(counts, edges[:-1], edges[1:]):
        x1 = scale(float(edge_left), x_domain, (left, right))
        x2 = scale(float(edge_right), x_domain, (left, right))
        y = scale(float(count), y_domain, (bottom, top))
        draw.rectangle((x1 + 1, y, x2 - 1, bottom), fill="#8DC7EA", outline=COLORS["background"], width=1)
    draw_polyline(draw, curve_x, curve_y, x_domain, y_domain, (left, right), (bottom, top), COLORS["blue"], width=5)
    draw.text(((left + right) // 2, 955), "Residuals", fill=COLORS["text"], font=find_font(23), anchor="mm")
    draw_vertical_label(image, "Frequency", (65, (top + bottom) // 2))
    save_image(image, "fig4_distribution_of_residuals.png")


def figure_term_importance(tests: Sequence[Mapping[str, object]]) -> None:
    image, draw = canvas(
        "Fig. 12 GAM Smooth-Term Contribution Ranking (Step 6)",
        "",
    )
    left, right = 510, 1650
    top, row_h = 145, 140
    draw.rectangle((left, 95, right, 880), fill=COLORS["panel"], outline=COLORS["grid"], width=2)
    maximum = max(float(row["drop_explained_deviance"]) for row in tests) * 1.12
    for percent in np.linspace(0.0, maximum, 7):
        x = scale(float(percent), (0.0, maximum), (left, right))
        draw.line((x, 95, x, 880), fill=COLORS["grid"], width=2)
        draw.text((x, 900), f"{percent:.1%}", fill=COLORS["text"], font=find_font(18), anchor="ma")
    for index, row in enumerate(tests):
        y = top + index * row_h
        value = float(row["drop_explained_deviance"])
        width = round(value / maximum * (right - left))
        draw.text((left - 32, y + 24), f"{row['rank']}. {row['smooth_term']}", fill=COLORS["text"], font=find_font(25), anchor="rm")
        draw.rectangle((left, y + 6, left + width, y + 50), fill=COLORS["blue"])
        draw.text((left + width + 18, y + 25), f"{value:.2%}", fill=COLORS["text"], font=find_font(22, bold=True), anchor="lm")
        draw.text((left, y + 60), f"EDF {float(row['effective_df']):.2f}   p={format_p_value(float(row['p_value']))} {row['significance']}", fill=COLORS["muted"], font=find_font(18))
    draw.text(((left + right) // 2, 960), "Drop in Explained Deviance", fill=COLORS["text"], font=find_font(23), anchor="mm")
    save_image(image, "fig5_smooth_term_ranking.png")


def save_image(image: Image.Image, filename: str) -> None:
    image.save(SCRIPT_DIR / filename, format="PNG", optimize=True)


def write_summary(
    n_rows: int,
    smoothing_lambda: float,
    fit: FitResult,
    baseline: FitResult,
    tests: Sequence[Mapping[str, object]],
) -> None:
    explained_deviance = fit.r2
    lines = [
        "=" * 80,
        "北京市 PM2.5 研究 - Step 6 GAM 实验结果说明",
        "=" * 80,
        "",
        "一、方法",
        f"- 使用变量/step6_gam.csv，共 {n_rows} 个完整日度样本。",
        "- 平滑项：WindSpeed、Humidity、NO2、CO、O3。",
        "- 线性控制项：Precipitation、Temperature、Lag_PM2.5、month。",
        f"- 使用惩罚三次 B 样条，每个平滑项 {N_BASIS} 个基函数；通过 GCV 选择平滑参数。",
        f"- 最优 lambda = {smoothing_lambda:.6g}。",
        "",
        "二、模型拟合",
        f"- GAM explained deviance：{explained_deviance:.2%}。",
        f"- GAM R²：{fit.r2:.4f}；调整 R²：{fit.adjusted_r2:.4f}。",
        f"- GAM RMSE：{fit.rmse:.4f}；MAE：{fit.mae:.4f}；AIC：{fit.aic:.2f}。",
        f"- 线性基线 R²：{baseline.r2:.4f}；RMSE：{baseline.rmse:.4f}；AIC：{baseline.aic:.2f}。",
        f"- 与线性基线相比，GAM 的 AIC 变化：{fit.aic - baseline.aic:+.2f}；R² 变化：{fit.r2 - baseline.r2:+.4f}。",
        "",
        "三、平滑项检验与贡献排序",
    ]
    for row in tests:
        lines.append(
            f"- {row['rank']}. {row['smooth_term']}：EDF = {float(row['effective_df']):.2f}，"
            f"F = {float(row['F_statistic']):.2f}，p = {format_p_value(float(row['p_value']))} "
            f"({row['significance']})，移除后解释偏差下降 {float(row['drop_explained_deviance']):.2%}。"
        )
    lines += [
        "",
        "四、解释提示",
        "- 平滑曲线表示在控制其他变量后，各核心变量变化对应的 PM2.5 边际效应。",
        "- 曲线明显弯曲或 EDF 高于约 1，说明单一直线难以完整描述该变量关系。",
        "- 本步骤用于检验关联关系中的非线性和阈值特征，不应直接表述为因果效应。",
        "- 可将本结果与贝叶斯网络和随机森林对照，讨论跨方法较稳定的关键变量。",
        "",
        "五、图表",
        "- fig1_gam_smooth_effects.png：五个核心变量的平滑边际效应及近似 95% 置信区间。",
        "- fig2_observed_vs_fitted.png：观测值与 GAM 拟合值对照。",
        "- fig3_residuals_vs_fitted.png：残差与拟合值诊断图。",
        "- fig4_distribution_of_residuals.png：残差分布直方图和平滑密度曲线。",
        "- fig5_smooth_term_ranking.png：移除单个平滑项后的解释偏差下降排序。",
    ]
    (SCRIPT_DIR / "实验结果说明.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure_notes() -> None:
    notes = """================================================================================
                    Step6 GAM 广义加性模型
--------------------------------------------------------------------------------
图1  fig1_gam_smooth_effects.png
--------------------------------------------------------------------------------
【图表类型】平滑边际效应曲线图
【画法 / 方法】对 WindSpeed、Humidity、NO2、CO、O3 使用惩罚三次 B 样条，
并绘制控制其他变量后的中心化边际效应和近似 95% 置信区间。
【目的】观察非线性、阈值、边际效应递减或 U 型关系。
【纵轴】相对平均水平的 PM2.5 边际变化量，单位与 PM2.5 相同。

--------------------------------------------------------------------------------
图2  fig2_observed_vs_fitted.png
--------------------------------------------------------------------------------
【图表类型】观测值与拟合值散点图
【画法 / 方法】横轴为 GAM 拟合 PM2.5，纵轴为实际 PM2.5；红线为 y=x。
【目的】评价模型拟合程度。散点越接近对角线，拟合误差越小。

--------------------------------------------------------------------------------
图3  fig3_residuals_vs_fitted.png
--------------------------------------------------------------------------------
【图表类型】残差诊断散点图
【画法 / 方法】横轴为 GAM 拟合值，纵轴为实际值减拟合值，红线为残差 0。
【目的】检查模型是否仍存在系统性偏差、异方差或极端误差。

--------------------------------------------------------------------------------
图4  fig4_distribution_of_residuals.png
--------------------------------------------------------------------------------
【图表类型】残差分布直方图和平滑密度曲线
【画法 / 方法】将 GAM 残差划分为 40 个等宽区间，柱高为频数；蓝线为核密度平滑后
按频数尺度换算的曲线。
【目的】观察残差是否集中在 0 附近，以及是否存在偏态或长尾。

--------------------------------------------------------------------------------
图5  fig5_smooth_term_ranking.png
--------------------------------------------------------------------------------
【图表类型】平滑项贡献排序图
【画法 / 方法】逐一移除平滑项，比较解释偏差下降幅度，并同时显示 EDF 与 p 值。
【目的】比较核心变量在 GAM 中的非线性解释贡献。
"""
    (SCRIPT_DIR / "图表说明文档.txt").write_text(notes, encoding="utf-8")


def main() -> None:
    dates, data = read_data(INPUT_CSV)
    y = data[TARGET]
    design, penalty, _, specs, _ = build_design(data)
    smoothing_lambda, fit, lambda_rows = select_lambda(design, y, penalty)
    baseline = fit_linear_baseline(data)
    tests = smooth_term_tests(design, y, penalty, fit, smoothing_lambda, specs)
    linear_rows = linear_term_rows(fit, data)
    partial_rows, curves = partial_effect_rows(fit, data, specs)

    metric_rows = (
        {"metric": "Sample rows", "value": len(y)},
        {"metric": "Smooth terms", "value": len(SMOOTH_TERMS)},
        {"metric": "Linear control terms", "value": len(LINEAR_TERMS)},
        {"metric": "Spline basis functions per smooth term", "value": N_BASIS},
        {"metric": "Selected lambda by GCV", "value": round(smoothing_lambda, 10)},
        {"metric": "Effective degrees of freedom", "value": round(fit.edf, 8)},
        {"metric": "Explained deviance", "value": round(fit.r2, 8)},
        {"metric": "R-squared", "value": round(fit.r2, 8)},
        {"metric": "Adjusted R-squared", "value": round(fit.adjusted_r2, 8)},
        {"metric": "RMSE", "value": round(fit.rmse, 8)},
        {"metric": "MAE", "value": round(fit.mae, 8)},
        {"metric": "AIC", "value": round(fit.aic, 8)},
        {"metric": "Linear baseline R-squared", "value": round(baseline.r2, 8)},
        {"metric": "Linear baseline RMSE", "value": round(baseline.rmse, 8)},
        {"metric": "Linear baseline AIC", "value": round(baseline.aic, 8)},
        {"metric": "GAM minus linear baseline AIC", "value": round(fit.aic - baseline.aic, 8)},
    )
    predictions = (
        {
            "date": date,
            "actual_PM2.5": round(float(actual), 8),
            "fitted_PM2.5": round(float(fitted), 8),
            "residual": round(float(residual), 8),
        }
        for date, actual, fitted, residual in zip(dates, y, fit.fitted, fit.residuals)
    )

    write_csv(SCRIPT_DIR / "gam_model_metrics.csv", ("metric", "value"), metric_rows)
    write_csv(SCRIPT_DIR / "gam_lambda_search.csv", ("lambda", "edf", "GCV", "AIC", "RMSE"), lambda_rows)
    write_csv(
        SCRIPT_DIR / "gam_smooth_terms.csv",
        ("rank", "smooth_term", "effective_df", "F_statistic", "p_value", "significance", "drop_RSS", "drop_explained_deviance"),
        tests,
    )
    write_csv(
        SCRIPT_DIR / "gam_linear_terms.csv",
        ("term", "coefficient", "std_error", "z_statistic_approx", "p_value_approx", "significance", "coefficient_scale"),
        linear_rows,
    )
    write_csv(SCRIPT_DIR / "gam_partial_effects.csv", ("smooth_term", "x", "partial_effect", "lower_95", "upper_95"), partial_rows)
    write_csv(SCRIPT_DIR / "gam_predictions.csv", ("date", "actual_PM2.5", "fitted_PM2.5", "residual"), predictions)

    metrics = {
        "r2": fit.r2,
        "adjusted_r2": fit.adjusted_r2,
        "explained_deviance": fit.r2,
        "rmse": fit.rmse,
        "mae": fit.mae,
        "aic": fit.aic,
    }
    figure_smooth_curves(curves, tests)
    figure_observed_vs_fitted(y, fit.fitted, metrics)
    figure_residuals(fit.fitted, fit.residuals)
    figure_residual_distribution(fit.residuals)
    figure_term_importance(tests)
    write_summary(len(y), smoothing_lambda, fit, baseline, tests)
    write_figure_notes()

    print(f"Completed GAM experiment with {len(y)} rows.")
    print(f"Selected lambda: {smoothing_lambda:.6g}")
    print(f"Explained deviance: {fit.r2:.2%}")
    print(f"RMSE: {fit.rmse:.4f}; AIC: {fit.aic:.2f}")
    print("Smooth-term contribution ranking:")
    for row in tests:
        print(
            f"  {row['rank']}. {row['smooth_term']}: "
            f"drop deviance={float(row['drop_explained_deviance']):.2%}, "
            f"EDF={float(row['effective_df']):.2f}, p={format_p_value(float(row['p_value']))}"
        )


if __name__ == "__main__":
    main()
