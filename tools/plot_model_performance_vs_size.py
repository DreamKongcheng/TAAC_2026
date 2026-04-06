from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = ROOT / "outputs" / "smoke"
DEFAULT_SIZE_OUTPUT_PATH = ROOT / "outputs" / "figures" / "model_performance_vs_size.png"
DEFAULT_COMPUTE_OUTPUT_PATH = ROOT / "outputs" / "figures" / "model_performance_vs_compute.png"

DISPLAY_NAMES = {
    "baseline": "baseline",
    "ctr_baseline": "ctr_baseline",
    "deepcontextnet": "deepcontextnet",
    "interformer": "interformer",
    "onetrans": "onetrans",
    "hyformer": "hyformer",
    "unirec": "unirec",
    "uniscaleformer": "uniscaleformer",
    "oo": "o_o",
}

LABEL_OFFSETS = {
    "baseline": (10, 8),
    "ctr_baseline": (10, -18),
    "deepcontextnet": (10, -18),
    "interformer": (10, -8),
    "onetrans": (10, 8),
    "hyformer": (10, -18),
    "unirec": (10, 10),
    "uniscaleformer": (10, -18),
    "oo": (10, -18),
}


@dataclass(slots=True)
class ModelPoint:
    slug: str
    label: str
    auc: float
    params_million: float
    parameter_size_mb: float
    total_tflops: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TAAC model performance against size or total training compute")
    parser.add_argument(
        "--summary-root",
        default=str(DEFAULT_SUMMARY_ROOT),
        help="Directory containing per-model smoke summary.json files",
    )
    parser.add_argument(
        "--x-metric",
        choices=("size", "compute"),
        default="size",
        help="Horizontal axis metric: model size in million parameters or estimated end-to-end training TFLOPs",
    )
    parser.add_argument(
        "--output-path",
        help="PNG output path; defaults to a metric-specific file name under outputs/figures",
    )
    return parser.parse_args()


def load_points(summary_root: Path) -> list[ModelPoint]:
    points: list[ModelPoint] = []
    missing: list[str] = []
    for slug, label in DISPLAY_NAMES.items():
        summary_path = summary_root / slug / "summary.json"
        if not summary_path.exists():
            missing.append(str(summary_path))
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        model_profile = payload["model_profile"]
        compute_profile = payload.get("compute_profile", {})
        points.append(
            ModelPoint(
                slug=slug,
                label=label,
                auc=float(payload["metrics"]["auc"]),
                params_million=float(model_profile["total_parameters"]) / 1.0e6,
                parameter_size_mb=float(model_profile["parameter_size_mb"]),
                total_tflops=float(compute_profile.get("estimated_end_to_end_tflops_total", 0.0)),
            )
        )
    if missing:
        missing_block = "\n".join(missing)
        raise FileNotFoundError(f"Missing summary files:\n{missing_block}")
    return points


def pareto_frontier(points: list[ModelPoint], x_getter) -> list[ModelPoint]:
    frontier: list[ModelPoint] = []
    best_auc = float("-inf")
    for point in sorted(points, key=x_getter):
        if point.auc > best_auc:
            frontier.append(point)
            best_auc = point.auc
    return frontier


def metric_config(x_metric: str) -> dict[str, object]:
    if x_metric == "compute":
        return {
            "title": "Model Performance VS Compute",
            "xlabel": "Estimated End-to-End Training Compute (TFLOPs)",
            "subtitle": "sample parquet, 10-epoch smoke",
            "output_path": DEFAULT_COMPUTE_OUTPUT_PATH,
            "x_getter": lambda point: point.total_tflops,
            "x_formatter": FuncFormatter(lambda value, _pos: f"{value:g}"),
            "x_scale": "log",
            "x_ticks": [0.1, 0.2, 0.5, 1, 2, 5, 10, 20],
            "highlight_note": "highlight = pareto frontier by lower training compute and higher AUC",
        }
    return {
        "title": "Model Performance VS Size",
        "xlabel": "Total Model Size (Million Parameters)",
        "subtitle": "sample parquet, 10-epoch smoke",
        "output_path": DEFAULT_SIZE_OUTPUT_PATH,
        "x_getter": lambda point: point.params_million,
        "x_formatter": FuncFormatter(lambda value, _pos: f"{value:.0f}"),
        "x_scale": "linear",
        "x_ticks": None,
        "highlight_note": "highlight = pareto frontier by smaller size and higher AUC",
    }


def render(points: list[ModelPoint], output_path: Path, x_metric: str) -> None:
    plt.style.use("dark_background")
    figure, axis = plt.subplots(figsize=(13, 9), facecolor="#111418")
    axis.set_facecolor("#111418")
    config = metric_config(x_metric)
    x_getter = config["x_getter"]

    frontier = pareto_frontier(points, x_getter)
    frontier_slugs = {point.slug for point in frontier}
    frontier = sorted(frontier, key=x_getter)

    other_points = [point for point in points if point.slug not in frontier_slugs]
    if other_points:
        axis.scatter(
            [x_getter(point) for point in other_points],
            [point.auc for point in other_points],
            s=110,
            color="#c7d0dc",
            edgecolors="none",
            alpha=0.95,
            zorder=3,
        )

    axis.scatter(
        [x_getter(point) for point in frontier],
        [point.auc for point in frontier],
        s=130,
        color="#3b82f6",
        edgecolors="#93c5fd",
        linewidths=1.5,
        alpha=1.0,
        zorder=4,
    )

    frontier_x = [x_getter(point) for point in frontier]
    frontier_y = [point.auc for point in frontier]
    fill_floor = min(point.auc for point in points) - 0.012
    axis.fill_between(frontier_x, frontier_y, [fill_floor] * len(frontier_x), color="#2563eb", alpha=0.22, zorder=1)
    axis.plot(frontier_x, frontier_y, color="#3b82f6", linewidth=2.5, alpha=0.95, zorder=2)

    for point in sorted(points, key=x_getter):
        dx, dy = LABEL_OFFSETS.get(point.slug, (10, 8))
        color = "#4f9cff" if point.slug in frontier_slugs else "#c7d0dc"
        axis.annotate(
            point.label,
            (x_getter(point), point.auc),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=12,
            color=color,
            alpha=0.98,
        )

    axis.set_title(str(config["subtitle"]), fontsize=13, color="#94a3b8", pad=12)
    figure.suptitle(str(config["title"]), fontsize=24, fontweight="bold", color="#f8fafc", y=0.965)
    axis.set_xlabel(str(config["xlabel"]), fontsize=15, color="#e5e7eb", labelpad=16)
    axis.set_ylabel("Validation AUC", fontsize=15, color="#e5e7eb", labelpad=14)

    axis.grid(True, color="#2b313c", linewidth=1.0, alpha=0.85)
    for spine in axis.spines.values():
        spine.set_color("#111418")

    axis.tick_params(axis="both", colors="#cbd5e1", labelsize=12)
    axis.set_xscale(str(config["x_scale"]))
    x_ticks = config["x_ticks"]
    if x_ticks is not None:
        tick_values = [tick for tick in x_ticks if min(x_getter(point) for point in points) * 0.8 <= tick <= max(x_getter(point) for point in points) * 1.2]
        axis.set_xticks(tick_values)
    axis.xaxis.set_major_formatter(config["x_formatter"])
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.3f}"))

    x_values = [x_getter(point) for point in points]
    axis.set_xlim(min(x_values) * 0.8, max(x_values) * 1.15)
    axis.set_ylim(fill_floor, max(point.auc for point in points) + 0.02)

    figure.text(
        0.995,
        0.018,
        str(config["highlight_note"]),
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="#94a3b8",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0.02, 0.04, 0.98, 0.94))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = metric_config(args.x_metric)
    output_path = Path(args.output_path) if args.output_path else Path(config["output_path"])
    render(load_points(Path(args.summary_root)), output_path, args.x_metric)


if __name__ == "__main__":
    main()