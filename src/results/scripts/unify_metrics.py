"""
Aggregate per-run metrics.csv files into an interactive HTML dashboard.

Expected input directory layout:
    <root_folder>/
      <attack_name>_<id>/metrics.csv
      <attack_name>_<id>/metrics.csv
      ...

Usage:
    python -m src.results.scripts.unify_metrics <root_folder>

The output HTML contains one subplot per metric and supports hiding attacks
through the legend. Axis ranges rescale according to visible attacks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate metrics.csv files from run subfolders into one interactive HTML dashboard"
    )
    parser.add_argument(
        "root_folder",
        type=str,
        help="Folder containing run subfolders like <attack_name>_<id>",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output HTML path. Defaults to <root_folder>/unified_metrics.html",
    )
    return parser.parse_args()


def split_attack_and_id(folder_name: str) -> tuple[str, str]:
    if "_" not in folder_name:
        return folder_name, ""
    attack_name, run_id = folder_name.split("_", 1)
    return attack_name, run_id


def load_metrics_row(metrics_path: Path) -> dict[str, object]:
    df = pd.read_csv(metrics_path)
    expected_columns = {"metric", "value"}
    if not expected_columns.issubset(df.columns):
        raise ValueError(
            f"{metrics_path} must contain columns {sorted(expected_columns)}, got {list(df.columns)}"
        )

    metric_series = df[["metric", "value"]].dropna(subset=["metric"]).set_index("metric")["value"]
    return metric_series.to_dict()


def create_metric_dashboard(unified_df: pd.DataFrame, output_path: Path) -> None:
    non_metric = {"attack_name", "run_folder"}
    metric_columns = [c for c in unified_df.columns if c not in non_metric]
    if not metric_columns:
        raise ValueError("No metric columns found to visualize")

    n_metrics = len(metric_columns)
    fig = make_subplots(
        rows=n_metrics,
        cols=1,
        shared_xaxes=False,
        subplot_titles=metric_columns,
        vertical_spacing=0.08,
    )

    attacks = sorted(unified_df["attack_name"].dropna().astype(str).unique())
    for row_idx, metric in enumerate(metric_columns, start=1):
        metric_df = unified_df[["attack_name", metric]].copy()
        metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
        metric_df = metric_df.dropna(subset=["attack_name", metric])
        attack_means = metric_df.groupby("attack_name", sort=True)[metric].mean()

        for attack in attacks:
            value = attack_means.get(attack)
            if pd.isna(value):
                continue

            fig.add_trace(
                go.Bar(
                    x=[attack],
                    y=[float(value)],
                    name=attack,
                    legendgroup=attack,
                    showlegend=(row_idx == 1),
                ),
                row=row_idx,
                col=1,
            )

        fig.update_yaxes(title_text="Score", row=row_idx, col=1, autorange=True, fixedrange=False)
        fig.update_xaxes(title_text="Attack", row=row_idx, col=1)

    fig.update_layout(
        title="Unified Metrics Dashboard",
        height=max(360 * n_metrics, 500),
        barmode="group",
        legend_title_text="Attack (click to hide/show)",
        legend={"groupclick": "togglegroup"},
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def main() -> int:
    args = parse_args()

    root = Path(args.root_folder)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root folder does not exist or is not a directory: {root}")

    rows: list[dict[str, object]] = []

    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        metrics_path = subdir / "metrics.csv"
        if not metrics_path.exists():
            continue

        attack_name, _ = split_attack_and_id(subdir.name)
        row: dict[str, object] = {
            "run_folder": subdir.name,
            "attack_name": attack_name,
        }
        row.update(load_metrics_row(metrics_path))
        rows.append(row)

    if not rows:
        raise ValueError(f"No metrics.csv files found in subfolders of: {root}")

    unified_df = pd.DataFrame(rows)

    ordered_cols = [
        *sorted(c for c in unified_df.columns if c not in {"attack_name", "run_folder"}),
        "attack_name",
        "run_folder",
    ]
    unified_df = unified_df[ordered_cols]

    output_path = Path(args.output) if args.output else root / "unified_metrics.html"
    create_metric_dashboard(unified_df=unified_df, output_path=output_path)

    non_metric = {"attack_name", "run_folder"}
    print(f"Wrote metric dashboard to: {output_path}")
    print(f"Rows aggregated: {len(unified_df)}")
    print(f"Metric panels: {len([c for c in unified_df.columns if c not in non_metric])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
