#!/usr/bin/env python3
"""Step 2: discrete Bayesian-network sensitivity analysis for daily PM2.5."""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_CSV = PROJECT_DIR / "变量" / "step2_bayesian_network.csv"

OUTCOME = "PM2.5_level"
OUTCOME_STATES = ("Low", "Medium", "High")
ALPHA = 1.0

PREDICTORS = (
    ("Temperature_level", "Temperature", ("Low", "Medium", "High"), "Meteorological"),
    ("Humidity_level", "Humidity", ("Low", "Medium", "High"), "Meteorological"),
    ("WindSpeed_level", "Wind speed", ("Low", "Medium", "High"), "Meteorological"),
    ("Precipitation_rain", "Precipitation", ("No rain", "Rain"), "Meteorological"),
    ("CO_level", "CO", ("Low", "Medium", "High"), "Gaseous pollutant"),
    ("NO2_level", "NO2", ("Low", "Medium", "High"), "Gaseous pollutant"),
    ("SO2_level", "SO2", ("Low", "Medium", "High"), "Gaseous pollutant"),
    ("O3_level", "O3", ("Low", "Medium", "High"), "Gaseous pollutant"),
    ("PM10_level", "PM10", ("Low", "Medium", "High"), "Particulate background"),
)

KEY_SCENARIOS = (
    ("Baseline", {}),
    ("Low wind speed", {"WindSpeed_level": "Low"}),
    ("High humidity", {"Humidity_level": "High"}),
    ("High NO2", {"NO2_level": "High"}),
    ("High CO", {"CO_level": "High"}),
    ("High CO + low wind speed", {"CO_level": "High", "WindSpeed_level": "Low"}),
)

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
    "low": "#D7ECF8",
    "medium": "#8DC7EA",
    "high": "#3498DB",
}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {OUTCOME, *(name for name, _, _, _ in PREDICTORS)}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class DiscreteNaiveBayesNetwork:
    """A discrete Bayesian network with PM2.5 level as the root class node."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.total = 0
        self.outcome_counts: Counter[str] = Counter()
        self.feature_counts: Dict[str, Counter[Tuple[str, str]]] = {}
        self.states = {name: states for name, _, states, _ in PREDICTORS}

    def fit(self, rows: Sequence[Mapping[str, str]]) -> None:
        self.total = len(rows)
        self.outcome_counts = Counter(row[OUTCOME] for row in rows)
        self.feature_counts = {name: Counter() for name, _, _, _ in PREDICTORS}
        for row in rows:
            outcome = row[OUTCOME]
            for feature, _, _, _ in PREDICTORS:
                self.feature_counts[feature][(outcome, row[feature])] += 1

    def prior(self, outcome: str) -> float:
        return (self.outcome_counts[outcome] + self.alpha) / (
            self.total + self.alpha * len(OUTCOME_STATES)
        )

    def likelihood(self, feature: str, state: str, outcome: str) -> float:
        state_count = len(self.states[feature])
        numerator = self.feature_counts[feature][(outcome, state)] + self.alpha
        denominator = self.outcome_counts[outcome] + self.alpha * state_count
        return numerator / denominator

    def posterior(self, evidence: Mapping[str, str]) -> Dict[str, float]:
        log_scores: Dict[str, float] = {}
        for outcome in OUTCOME_STATES:
            score = math.log(self.prior(outcome))
            for feature, state in evidence.items():
                score += math.log(self.likelihood(feature, state, outcome))
            log_scores[outcome] = score
        peak = max(log_scores.values())
        weights = {state: math.exp(score - peak) for state, score in log_scores.items()}
        total = sum(weights.values())
        return {state: weight / total for state, weight in weights.items()}

    def high_probability(self, evidence: Mapping[str, str]) -> float:
        return self.posterior(evidence)["High"]


def empirical_high_probability(rows: Sequence[Mapping[str, str]], evidence: Mapping[str, str]) -> Tuple[int, float]:
    subset = [row for row in rows if all(row[key] == value for key, value in evidence.items())]
    if not subset:
        return 0, float("nan")
    high_count = sum(row[OUTCOME] == "High" for row in subset)
    return len(subset), high_count / len(subset)


def build_results(rows: Sequence[Mapping[str, str]], model: DiscreteNaiveBayesNetwork):
    base_empirical = sum(row[OUTCOME] == "High" for row in rows) / len(rows)
    base_bn = model.high_probability({})

    cpt_rows = []
    for feature, label, states, category in PREDICTORS:
        for outcome in OUTCOME_STATES:
            for state in states:
                cpt_rows.append(
                    {
                        "parent_PM2.5_level": outcome,
                        "node": feature,
                        "node_label": label,
                        "category": category,
                        "state": state,
                        "count": model.feature_counts[feature][(outcome, state)],
                        "conditional_probability": round(model.likelihood(feature, state, outcome), 8),
                    }
                )

    sensitivity_rows = []
    for feature, label, states, category in PREDICTORS:
        for state in states:
            evidence = {feature: state}
            n_days, empirical = empirical_high_probability(rows, evidence)
            bn_probability = model.high_probability(evidence)
            sensitivity_rows.append(
                {
                    "variable": feature,
                    "variable_label": label,
                    "category": category,
                    "state": state,
                    "n_days": n_days,
                    "empirical_P_high": round(empirical, 8),
                    "BN_P_high": round(bn_probability, 8),
                    "delta_P": round(bn_probability - base_bn, 8),
                    "abs_delta_P": round(abs(bn_probability - base_bn), 8),
                }
            )

    ranking_rows = []
    for feature, label, _, category in PREDICTORS:
        feature_rows = [row for row in sensitivity_rows if row["variable"] == feature]
        strongest = max(feature_rows, key=lambda row: row["abs_delta_P"])
        lowest = min(feature_rows, key=lambda row: row["BN_P_high"])
        highest = max(feature_rows, key=lambda row: row["BN_P_high"])
        ranking_rows.append(
            {
                "variable": feature,
                "variable_label": label,
                "category": category,
                "strongest_state": strongest["state"],
                "strongest_delta_P": strongest["delta_P"],
                "max_abs_delta_P": strongest["abs_delta_P"],
                "lowest_risk_state": lowest["state"],
                "lowest_BN_P_high": lowest["BN_P_high"],
                "highest_risk_state": highest["state"],
                "highest_BN_P_high": highest["BN_P_high"],
                "probability_range": round(highest["BN_P_high"] - lowest["BN_P_high"], 8),
            }
        )
    ranking_rows.sort(key=lambda row: row["max_abs_delta_P"], reverse=True)
    for index, row in enumerate(ranking_rows, start=1):
        row["rank"] = index

    scenario_rows = []
    for label, evidence in KEY_SCENARIOS:
        n_days, empirical = empirical_high_probability(rows, evidence)
        probability = model.high_probability(evidence)
        scenario_rows.append(
            {
                "scenario": label,
                "evidence": "; ".join(f"{key}={value}" for key, value in evidence.items()) or "None",
                "matching_days": n_days,
                "empirical_P_high": round(empirical, 8),
                "BN_P_high": round(probability, 8),
                "delta_P": round(probability - base_bn, 8),
            }
        )

    return base_empirical, base_bn, cpt_rows, sensitivity_rows, ranking_rows, scenario_rows


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


def save_image(image: Image.Image, name: str) -> None:
    image.save(SCRIPT_DIR / name, format="PNG", optimize=True)


def draw_arrow(draw: ImageDraw.ImageDraw, start: Tuple[int, int], end: Tuple[int, int], color: str) -> None:
    draw.line((*start, *end), fill=color, width=4)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_len = 18
    for offset in (2.55, -2.55):
        point = (
            end[0] + arrow_len * math.cos(angle + offset),
            end[1] + arrow_len * math.sin(angle + offset),
        )
        draw.line((*end, *point), fill=color, width=4)


def draw_node(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], label: str, fill: str, outline: str) -> None:
    draw.rounded_rectangle(box, radius=8, fill=fill, outline=outline, width=3)
    draw.text(
        ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
        label,
        fill=COLORS["text"],
        font=find_font(23, bold=True),
        anchor="mm",
    )


def figure_network() -> None:
    image, draw = canvas(
        "Fig. 1 Bayesian Network Structure (Step 2)",
        "",
    )
    root = (680, 145, 1120, 245)
    node_boxes = []
    positions = [
        (120, 395), (455, 395), (790, 395), (1125, 395), (1460, 395),
        (285, 665), (620, 665), (955, 665), (1290, 665),
    ]
    for (_, label, _, category), (x, y) in zip(PREDICTORS, positions):
        node_boxes.append((x, y, x + 220, y + 82, label, category))
    for x1, y1, x2, y2, _, _ in node_boxes:
        draw_arrow(draw, (900, root[3]), ((x1 + x2) // 2, y1), "#8A8A8A")
    draw_node(draw, root, "PM2.5 level", "#8DC7EA", "#2C7FB8")
    fills = {"Meteorological": "#D7ECF8", "Gaseous pollutant": "#8DC7EA", "Particulate background": "#3498DB"}
    outlines = {"Meteorological": "#2C7FB8", "Gaseous pollutant": "#2C7FB8", "Particulate background": "#2C7FB8"}
    for x1, y1, x2, y2, label, category in node_boxes:
        draw_node(draw, (x1, y1, x2, y2), label, fills[category], outlines[category])
    legend = [("Meteorological", "#D7ECF8"), ("Gaseous pollutant", "#8DC7EA"), ("Particulate background", "#3498DB")]
    x = 505
    for label, color in legend:
        draw.rectangle((x, 865, x + 26, 891), fill=color, outline="#2C7FB8", width=1)
        draw.text((x + 38, 864), label, fill=COLORS["text"], font=find_font(20))
        x += 315
    draw.text(
        (900, 945),
        "Arrows indicate the generative model factorization for posterior inference, not causal effects.",
        fill=COLORS["muted"],
        font=find_font(19),
        anchor="mm",
    )
    save_image(image, "fig1_bayesian_network_structure.png")


def probability_color(probability: float, baseline: float) -> str:
    strength = min(max(probability, 0.0), 1.0)
    target = (52, 152, 219)
    base = (250, 250, 250)
    rgb = tuple(round(base[i] + strength * (target[i] - base[i])) for i in range(3))
    return "#%02X%02X%02X" % rgb


def figure_probability_matrix(sensitivity_rows, baseline: float) -> None:
    image, draw = canvas(
        "Fig. 2 Posterior Probability of High PM2.5 by Observed State (Step 2)",
        "",
    )
    row_lookup = {(row["variable"], row["state"]): row for row in sensitivity_rows}
    x0, y0 = 590, 120
    cell_w, cell_h = 260, 76
    draw.rectangle((x0 - 300, y0 - 30, x0 + 3 * cell_w + 10, y0 + 55 + len(PREDICTORS) * cell_h + 10), fill=COLORS["panel"], outline=COLORS["grid"], width=2)
    column_states = ("State 1", "State 2", "State 3")
    for index, state in enumerate(column_states):
        draw.text((x0 + index * cell_w + cell_w // 2, y0), state, fill=COLORS["text"], font=find_font(23, bold=True), anchor="mm")
    for row_index, (feature, label, states, _) in enumerate(PREDICTORS):
        y = y0 + 55 + row_index * cell_h
        draw.text((x0 - 35, y + cell_h // 2), label, fill=COLORS["text"], font=find_font(23), anchor="rm")
        display_states = states if feature != "Precipitation_rain" else ("No rain", "Rain")
        for column_index in range(3):
            x = x0 + column_index * cell_w
            box = (x + 5, y + 5, x + cell_w - 7, y + cell_h - 7)
            if column_index >= len(display_states):
                draw.rectangle(box, fill="#F2F4F6", outline=COLORS["grid"], width=2)
                draw.text((x + cell_w // 2, y + cell_h // 2), "n/a", fill=COLORS["muted"], font=find_font(19), anchor="mm")
                continue
            state = display_states[column_index]
            probability = float(row_lookup[(feature, state)]["BN_P_high"])
            draw.rectangle(box, fill=probability_color(probability, baseline), outline=COLORS["grid"], width=2)
            text_fill = COLORS["background"] if probability > 0.62 else COLORS["text"]
            draw.text((x + cell_w // 2, y + 23), state, fill=text_fill, font=find_font(16), anchor="mm")
            draw.text((x + cell_w // 2, y + 49), f"{probability:.1%}", fill=text_fill, font=find_font(21, bold=True), anchor="mm")
    draw.text((360, 945), f"Baseline P(PM2.5 = High): {baseline:.1%}", fill=COLORS["text"], font=find_font(21))
    draw.rectangle((1180, 930, 1260, 953), fill=probability_color(0.05, baseline), outline=COLORS["grid"], width=1)
    draw.text((1270, 930), "Lower risk", fill=COLORS["text"], font=find_font(19))
    draw.rectangle((1430, 930, 1510, 953), fill=probability_color(0.85, baseline), outline=COLORS["grid"], width=1)
    draw.text((1520, 930), "Higher risk", fill=COLORS["text"], font=find_font(19))
    save_image(image, "fig2_high_pollution_probability_by_state.png")


def figure_sensitivity_ranking(ranking_rows) -> None:
    image, draw = canvas(
        "Fig. 3 Sensitivity Ranking for High PM2.5 Probability (Step 2)",
        "",
    )
    left, right = 470, 1690
    top, row_h = 125, 82
    max_abs = max(abs(float(row["strongest_delta_P"])) for row in ranking_rows) * 1.15
    zero_x = (left + right) // 2
    draw.rectangle((left, 90, right, 885), fill=COLORS["panel"], outline=COLORS["grid"], width=2)
    for tick in (max_abs * (index - 4) / 4 for index in range(9)):
        x = round(zero_x + tick / max_abs * ((right - left) / 2))
        draw.line((x, 90, x, 885), fill=COLORS["grid"], width=2)
        draw.text((x, 905), f"{tick:+.0%}", fill=COLORS["text"], font=find_font(18), anchor="ma")
    draw_dashed_line(draw, (zero_x, 90), (zero_x, 885), fill=COLORS["coral"], width=5)
    for index, row in enumerate(ranking_rows):
        y = top + index * row_h
        delta = float(row["strongest_delta_P"])
        width = int(abs(delta) / max_abs * ((right - left) / 2))
        x1, x2 = (zero_x, zero_x + width) if delta >= 0 else (zero_x - width, zero_x)
        draw.rectangle((x1, y + 10, x2, y + 52), fill=COLORS["blue"])
        draw.text((left - 35, y + 28), f"{row['rank']}. {row['variable_label']}", fill=COLORS["text"], font=find_font(23), anchor="rm")
        label = f"{row['strongest_state']}: {delta:+.1%}"
        label_x = x2 + 16 if delta >= 0 else x1 - 16
        anchor = "lm" if delta >= 0 else "rm"
        draw.text((label_x, y + 28), label, fill=COLORS["text"], font=find_font(21, bold=True), anchor=anchor)
    draw.text(((left + right) // 2, 970), "Delta P Relative to Baseline", fill=COLORS["text"], font=find_font(23), anchor="mm")
    save_image(image, "fig3_sensitivity_ranking.png")


def figure_key_scenarios(scenario_rows, baseline: float) -> None:
    image, draw = canvas(
        "Fig. 4 High PM2.5 Probability in Key Scenarios (Step 2)",
        "",
    )
    left, right = 590, 1640
    top, row_h = 160, 115
    draw.rectangle((left, 100, right, 865), fill=COLORS["panel"], outline=COLORS["grid"], width=2)
    for percent in range(0, 101, 10):
        x = left + int(percent / 100 * (right - left))
        draw.line((x, 100, x, 865), fill=COLORS["grid"], width=2)
        draw.text((x, 885), f"{percent}%", fill=COLORS["text"], font=find_font(18), anchor="ma")
    baseline_x = left + int(baseline * (right - left))
    draw_dashed_line(draw, (baseline_x, 100), (baseline_x, 865), fill=COLORS["coral"], width=5)
    draw.text((baseline_x, 72), f"Baseline {baseline:.1%}", fill=COLORS["coral"], font=find_font(20, bold=True), anchor="mm")
    for index, row in enumerate(scenario_rows):
        y = top + index * row_h
        probability = float(row["BN_P_high"])
        bar_right = left + int(probability * (right - left))
        draw.text((left - 30, y + 23), row["scenario"], fill=COLORS["text"], font=find_font(23), anchor="rm")
        draw.rectangle((left, y + 5, bar_right, y + 51), fill=COLORS["blue"])
        draw.text((bar_right + 18, y + 24), f"{probability:.1%}", fill=COLORS["text"], font=find_font(23, bold=True), anchor="lm")
    draw.text(((left + right) // 2, 960), "Posterior Probability of High PM2.5", fill=COLORS["text"], font=find_font(23), anchor="mm")
    save_image(image, "fig4_key_scenarios.png")


def write_summary(rows, base_empirical, base_bn, ranking_rows, scenario_rows) -> None:
    top = ranking_rows[:5]
    scenario_lookup = {row["scenario"]: row for row in scenario_rows}
    lines = [
        "=" * 80,
        "北京市 PM2.5 研究 - Step 2 贝叶斯网络实验结果说明",
        "=" * 80,
        "",
        "一、方法",
        f"- 使用变量/step2_bayesian_network.csv，共 {len(rows)} 个日度样本。",
        "- 网络采用离散朴素贝叶斯结构：PM2.5_level 为根节点，各气象、气态污染物和 PM10 档位为观测节点。",
        "- 使用拉普拉斯平滑（alpha=1）估计条件概率，避免稀疏组合造成概率为 0。",
        "- 箭头用于表达联合分布的分解和后验推断，不应解释为因果方向。",
        "- PM2.5_high 是由 PM2.5_level 派生的标记，仅用于核对，不进入网络，避免信息泄漏。",
        "",
        "二、基准概率",
        f"- 样本中高污染档占比：{base_empirical:.2%}。",
        f"- 贝叶斯网络平滑后的 P(PM2.5 = High)：{base_bn:.2%}。",
        "",
        "三、敏感性排序（按最大 |Delta P|）",
    ]
    for row in top:
        lines.append(
            f"- {row['rank']}. {row['variable_label']}：最敏感状态为 {row['strongest_state']}，"
            f"Delta P = {float(row['strongest_delta_P']):+.2%}；"
            f"高风险状态为 {row['highest_risk_state']}，P(High) = {float(row['highest_BN_P_high']):.2%}。"
        )
    lines += [
        "",
        "四、研究方案要求的重点概率",
    ]
    for scenario in KEY_SCENARIOS:
        row = scenario_lookup[scenario[0]]
        lines.append(
            f"- {row['scenario']}：P(High) = {float(row['BN_P_high']):.2%}，"
            f"Delta P = {float(row['delta_P']):+.2%}。"
        )
    lines += [
        "",
        "五、报告使用提示",
        "- 单因素敏感性结论来自概率关联，不能直接写成因果效应。",
        "- PM10 与 PM2.5 同属颗粒污染指标。若 PM10 排名靠前，应作为污染背景变量单独说明，",
        "  不宜将其与温度、湿度、风速或气态污染物作完全相同的机制解释。",
        "- 可将本步骤排序与随机森林特征重要性对照，讨论跨方法较稳定的关键变量。",
        "",
        "六、图表",
        "- fig1_bayesian_network_structure.png：贝叶斯网络结构图。",
        "- fig2_high_pollution_probability_by_state.png：各变量状态下高污染后验概率矩阵。",
        "- fig3_sensitivity_ranking.png：按最大 |Delta P| 排序的敏感性图。",
        "- fig4_key_scenarios.png：研究方案重点情景概率图。",
    ]
    (SCRIPT_DIR / "实验结果说明.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure_notes() -> None:
    notes = """================================================================================
                    Step2 贝叶斯网络敏感性分析
--------------------------------------------------------------------------------
图1  fig1_bayesian_network_structure.png
--------------------------------------------------------------------------------
【图表类型】贝叶斯网络结构图
【画法 / 方法】PM2.5_level 作为根节点，离散化气象变量、气态污染物和 PM10 作为观测节点。
【目的】展示模型用于概率推断时的联合分布分解结构。
【注意】箭头表示模型分解方向，不是因果方向。

--------------------------------------------------------------------------------
图2  fig2_high_pollution_probability_by_state.png
--------------------------------------------------------------------------------
【图表类型】条件概率矩阵图
【画法 / 方法】逐一将各变量设为 Low、Medium、High（降水为 No rain / Rain），
计算贝叶斯网络后验概率 P(PM2.5 = High | X = x)。蓝色越深表示高污染概率越高。
【目的】直观比较不同观测状态对应的高污染概率。

--------------------------------------------------------------------------------
图3  fig3_sensitivity_ranking.png
--------------------------------------------------------------------------------
【图表类型】水平条形排序图
【画法 / 方法】对每个变量选取 |Delta P| 最大的状态，其中
Delta P = P(PM2.5 = High | X = x) - P(PM2.5 = High)。
【目的】回答哪些变量状态最明显地改变高污染概率。

--------------------------------------------------------------------------------
图4  fig4_key_scenarios.png
--------------------------------------------------------------------------------
【图表类型】重点情景概率条形图
【画法 / 方法】比较基准情景、低风速、高湿度、高 NO2、高 CO、
高 CO + 低风速下的高污染后验概率。
【目的】对应研究方案指定的重点分析问题，便于直接写入报告。
"""
    (SCRIPT_DIR / "图表说明文档.txt").write_text(notes, encoding="utf-8")


def main() -> None:
    rows = read_rows(INPUT_CSV)
    model = DiscreteNaiveBayesNetwork(alpha=ALPHA)
    model.fit(rows)
    base_empirical, base_bn, cpt_rows, sensitivity_rows, ranking_rows, scenario_rows = build_results(rows, model)

    write_csv(
        SCRIPT_DIR / "bn_network_edges.csv",
        ("from_node", "to_node", "interpretation"),
        (
            {"from_node": OUTCOME, "to_node": feature, "interpretation": "Generative model edge; not a causal claim"}
            for feature, _, _, _ in PREDICTORS
        ),
    )
    write_csv(
        SCRIPT_DIR / "bn_cpt.csv",
        ("parent_PM2.5_level", "node", "node_label", "category", "state", "count", "conditional_probability"),
        cpt_rows,
    )
    write_csv(
        SCRIPT_DIR / "bn_sensitivity_by_state.csv",
        ("variable", "variable_label", "category", "state", "n_days", "empirical_P_high", "BN_P_high", "delta_P", "abs_delta_P"),
        sensitivity_rows,
    )
    write_csv(
        SCRIPT_DIR / "bn_sensitivity_ranking.csv",
        (
            "rank", "variable", "variable_label", "category", "strongest_state", "strongest_delta_P",
            "max_abs_delta_P", "lowest_risk_state", "lowest_BN_P_high", "highest_risk_state",
            "highest_BN_P_high", "probability_range",
        ),
        ranking_rows,
    )
    write_csv(
        SCRIPT_DIR / "bn_key_scenarios.csv",
        ("scenario", "evidence", "matching_days", "empirical_P_high", "BN_P_high", "delta_P"),
        scenario_rows,
    )
    write_csv(
        SCRIPT_DIR / "bn_model_summary.csv",
        ("metric", "value"),
        (
            {"metric": "Sample rows", "value": len(rows)},
            {"metric": "Predictor nodes", "value": len(PREDICTORS)},
            {"metric": "Network edges", "value": len(PREDICTORS)},
            {"metric": "Laplace smoothing alpha", "value": ALPHA},
            {"metric": "Empirical baseline P(PM2.5 = High)", "value": round(base_empirical, 8)},
            {"metric": "BN baseline P(PM2.5 = High)", "value": round(base_bn, 8)},
        ),
    )

    figure_network()
    figure_probability_matrix(sensitivity_rows, base_bn)
    figure_sensitivity_ranking(ranking_rows)
    figure_key_scenarios(scenario_rows, base_bn)
    write_summary(rows, base_empirical, base_bn, ranking_rows, scenario_rows)
    write_figure_notes()

    print(f"Completed Bayesian-network experiment with {len(rows)} rows.")
    print(f"Baseline P(PM2.5 = High): {base_bn:.2%}")
    print("Top sensitivity variables:")
    for row in ranking_rows[:5]:
        print(f"  {row['rank']}. {row['variable_label']}: {row['strongest_state']} ({float(row['strongest_delta_P']):+.2%})")


if __name__ == "__main__":
    main()
