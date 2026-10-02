#!/usr/bin/env python3
"""
Final six-model comparison figures: holdout parity plot across all six models (01), Taylor
diagram (02), and the XGBoost training-vs-holdout parity plot (03). Reads the saved
predictions and metrics from model_development/fit_6models_one_final.py.
"""

import os
from pathlib import Path
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from scipy.ndimage import gaussian_filter
from scipy.stats import linregress


# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = Path(os.environ.get("AQUAMATCH_DATA_DIR", "./data"))
RESULTS_DIR = Path(os.environ.get("AQUAMATCH_RESULTS_DIR", "./results"))

RESULT_ROOT = DATA_DIR / "model_comparison_6models_final"
MODEL_ROOT = RESULT_ROOT / "models"
PLOT_DIR = RESULTS_DIR / "figures"

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# SETTINGS
# =============================================================================

DPI = 600

PARITY_MIN = 0.05
PARITY_MAX = 300.0

# 0.05 is the lower plotting boundary but is deliberately not labelled.
PARITY_TICKS = [
    0.1,
    1.0,
    10.0,
    100.0,
]

PARITY_TICK_LABELS = [
    "0.1",
    "1",
    "10",
    "100",
]


MODELS = [
    ("pls",     "PLS"),
    ("kpls",    "Kernel PLS"),
    ("mlp",     "MLP"),
    ("realmlp", "RealMLP"),
    ("rf",      "Random Forest"),
    ("xgb",     "XGBoost"),
]


# =============================================================================
# LOAD SAVED PREDICTIONS
# =============================================================================

def load_predictions():

    loaded = {}

    for key, name in MODELS:

        model_dir = MODEL_ROOT / key

        train_path = model_dir / "predictions_train.csv"
        holdout_path = model_dir / "predictions_holdout.csv"
        metrics_path = model_dir / "metrics.json"

        for path in [
            train_path,
            holdout_path,
            metrics_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing required file for {name}:\n{path}"
                )

        train = pd.read_csv(train_path)
        holdout = pd.read_csv(holdout_path)

        metadata = json.loads(
            metrics_path.read_text()
        )

        for subset_name, frame in [
            ("training", train),
            ("holdout", holdout),
        ]:

            required = [
                "observed_chl_a",
                "predicted_chl_a",
            ]

            missing = [
                c
                for c in required
                if c not in frame.columns
            ]

            if missing:
                raise KeyError(
                    f"{name} {subset_name} file is missing "
                    f"{missing}.\nColumns: {list(frame.columns)}"
                )

        loaded[key] = {
            "name": name,
            "train": train,
            "holdout": holdout,
            "metadata": metadata,
        }

        print(
            f"{name:15s} | "
            f"training n={len(train):,} | "
            f"holdout n={len(holdout):,}"
        )

    return loaded


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(observed, predicted):

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    finite = (
        np.isfinite(observed)
        &
        np.isfinite(predicted)
    )

    observed = observed[finite]
    predicted = predicted[finite]

    residual = (
        predicted
        -
        observed
    )

    ss_res = np.sum(
        residual ** 2
    )

    ss_tot = np.sum(
        (
            observed
            -
            np.mean(observed)
        ) ** 2
    )

    r2 = 1.0 - ss_res / ss_tot

    rmse = np.sqrt(
        np.mean(
            residual ** 2
        )
    )

    mae = np.mean(
        np.abs(
            residual
        )
    )

    bias = np.mean(
        residual
    )

    fit = linregress(
        observed,
        predicted,
    )

    return {
        "n": int(len(observed)),
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "bias": float(bias),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "pearson_r": float(fit.rvalue),
    }


# =============================================================================
# LOCAL POINT DENSITY
# =============================================================================

def relative_density(
    observed,
    predicted,
    bins=80,
    sigma=1.15,
):

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    # Plotting/density values only.
    xp = np.clip(
        observed,
        PARITY_MIN,
        PARITY_MAX,
    )

    yp = np.clip(
        predicted,
        PARITY_MIN,
        PARITY_MAX,
    )

    lx = np.log10(xp)
    ly = np.log10(yp)

    lo = np.log10(PARITY_MIN)
    hi = np.log10(PARITY_MAX)

    H, xedges, yedges = np.histogram2d(
        lx,
        ly,
        bins=bins,
        range=[
            [lo, hi],
            [lo, hi],
        ],
    )

    H = gaussian_filter(
        H.astype(float),
        sigma=sigma,
    )

    ix = np.clip(
        np.searchsorted(
            xedges,
            lx,
            side="right",
        ) - 1,
        0,
        H.shape[0] - 1,
    )

    iy = np.clip(
        np.searchsorted(
            yedges,
            ly,
            side="right",
        ) - 1,
        0,
        H.shape[1] - 1,
    )

    density = H[ix, iy]

    density = np.log1p(
        density
    )

    if np.nanmax(density) > np.nanmin(density):

        density = (
            density
            -
            np.nanmin(density)
        ) / (
            np.nanmax(density)
            -
            np.nanmin(density)
        )

    else:

        density = np.zeros_like(
            density
        )

    return density


# =============================================================================
# PARITY AXIS
# =============================================================================

def format_parity_axis(ax):

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(
        PARITY_MIN,
        PARITY_MAX,
    )

    ax.set_ylim(
        PARITY_MIN,
        PARITY_MAX,
    )

    ax.set_xticks(
        PARITY_TICKS
    )

    ax.set_yticks(
        PARITY_TICKS
    )

    ax.set_xticklabels(
        PARITY_TICK_LABELS
    )

    ax.set_yticklabels(
        PARITY_TICK_LABELS
    )

    # Keep every panel visually square.
    ax.set_box_aspect(1.0)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=12,
        length=5,
        width=1.0,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        length=2.5,
        width=0.7,
    )

    ax.grid(
        True,
        which="major",
        linewidth=0.7,
        alpha=0.18,
    )

    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.45,
        alpha=0.07,
    )


# =============================================================================
# PARITY PANEL
# =============================================================================

def plot_parity_panel(
    ax,
    observed,
    predicted,
    title,
):

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    metrics = calculate_metrics(
        observed,
        predicted,
    )

    observed_plot = np.clip(
        observed,
        PARITY_MIN,
        PARITY_MAX,
    )

    predicted_plot = np.clip(
        predicted,
        PARITY_MIN,
        PARITY_MAX,
    )

    density = relative_density(
        observed,
        predicted,
    )

    # Low-density points underneath dense points.
    order = np.argsort(
        density
    )

    scatter = ax.scatter(
        observed_plot[order],
        predicted_plot[order],
        c=density[order],
        cmap="viridis",
        norm=Normalize(0, 1),
        s=15,
        alpha=0.80,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )


    # -------------------------------------------------------------------------
    # 1:1 reference
    # -------------------------------------------------------------------------

    ax.plot(
        [
            PARITY_MIN,
            PARITY_MAX,
        ],
        [
            PARITY_MIN,
            PARITY_MAX,
        ],
        linestyle="--",
        color="black",
        linewidth=1.45,
        alpha=0.80,
        label="1:1",
        zorder=4,
    )


    # -------------------------------------------------------------------------
    # Ordinary least-squares fit calculated on ORIGINAL Chl-a values.
    # -------------------------------------------------------------------------

    slope = metrics["slope"]
    intercept = metrics["intercept"]

    x_fit = np.logspace(
        np.log10(PARITY_MIN),
        np.log10(PARITY_MAX),
        600,
    )

    y_fit = (
        slope
        *
        x_fit
        +
        intercept
    )

    visible = (
        y_fit > 0
    )

    ax.plot(
        x_fit[visible],
        y_fit[visible],
        color="#C62828",
        linewidth=1.65,
        label=(
            f"Linear Fit: y = {slope:.2f}x"
            f"{intercept:+.1f}"
        ),
        zorder=5,
    )


    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metric_text = (
        f"n = {metrics['n']:,}\n"
        f"R² = {metrics['r2']:.3f}\n"
        f"RMSE = {metrics['rmse']:.2f} µg/L\n"
        f"MAE = {metrics['mae']:.2f} µg/L\n"
        f"Bias = {metrics['bias']:+.2f} µg/L"
    )

    ax.text(
        0.035,
        0.965,
        metric_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.8,
        linespacing=1.16,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="0.55",
            linewidth=0.8,
            alpha=0.94,
        ),
        zorder=10,
    )


    format_parity_axis(
        ax
    )

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=7,
    )

    ax.set_xlabel(
        "Measured Chl-a (µg/L)",
        fontsize=14,
        labelpad=7,
    )

    ax.set_ylabel(
        "Predicted Chl-a (µg/L)",
        fontsize=14,
        labelpad=7,
    )

    ax.legend(
        loc="lower right",
        fontsize=9.5,
        frameon=True,
        framealpha=0.95,
        borderpad=0.55,
        labelspacing=0.36,
        handlelength=2.0,
    )

    return scatter, metrics


# =============================================================================
# SIX-MODEL HOLDOUT FIGURE
# =============================================================================

def make_six_model_figure(loaded):

    fig = plt.figure(
        figsize=(
            24.0,
            12.6,
        )
    )

    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[
            1.0,
            1.0,
            1.0,
            0.035,
        ],
        wspace=0.28,
        hspace=0.30,
    )

    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    ]

    cax = fig.add_subplot(
        grid[:, 3]
    )

    metric_rows = []
    scatter = None

    for ax, (
        key,
        name,
    ) in zip(
        axes,
        MODELS,
    ):

        holdout = loaded[
            key
        ][
            "holdout"
        ]

        scatter, metrics = plot_parity_panel(
            ax,
            holdout["observed_chl_a"],
            holdout["predicted_chl_a"],
            name,
        )

        metric_rows.append({
            "model": name,
            **metrics,
        })

    # Keep labels on top row.
    for ax in axes:

        ax.tick_params(
            axis="x",
            which="both",
            labelbottom=True,
        )

        ax.tick_params(
            axis="y",
            which="both",
            labelleft=True,
        )

    colorbar = fig.colorbar(
        scatter,
        cax=cax,
    )

    colorbar.set_label(
        "Relative point density",
        fontsize=17,
        labelpad=12,
    )

    colorbar.ax.tick_params(
        labelsize=12.5,
    )

    # No overall title.

    png = (
        PLOT_DIR
        / "01_six_model_holdout_parity.png"
    )

    pdf = (
        PLOT_DIR
        / "01_six_model_holdout_parity.pdf"
    )

    svg = (
        PLOT_DIR
        / "01_six_model_holdout_parity.svg"
    )

    fig.savefig(
        png,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        svg,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_df.to_csv(
        PLOT_DIR
        / "01_six_model_holdout_metrics.csv",
        index=False,
    )

    return metrics_df


# =============================================================================
# XGBOOST TRAINING + HOLDOUT
# =============================================================================

def make_xgboost_training_holdout(loaded):

    train = loaded[
        "xgb"
    ][
        "train"
    ]

    holdout = loaded[
        "xgb"
    ][
        "holdout"
    ]

    fig = plt.figure(
        figsize=(
            16.5,
            7.0,
        )
    )

    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[
            1.0,
            1.0,
            0.035,
        ],
        wspace=0.27,
    )

    ax_train = fig.add_subplot(
        grid[0, 0]
    )

    ax_holdout = fig.add_subplot(
        grid[0, 1]
    )

    cax = fig.add_subplot(
        grid[0, 2]
    )

    scatter, train_metrics = plot_parity_panel(
        ax_train,
        train["observed_chl_a"],
        train["predicted_chl_a"],
        "(a) Training",
    )

    scatter, holdout_metrics = plot_parity_panel(
        ax_holdout,
        holdout["observed_chl_a"],
        holdout["predicted_chl_a"],
        "(b) Holdout test",
    )

    fig.suptitle(
        "XGBoost",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )

    colorbar = fig.colorbar(
        scatter,
        cax=cax,
    )

    colorbar.set_label(
        "Relative point density",
        fontsize=15.5,
        labelpad=11,
    )

    colorbar.ax.tick_params(
        labelsize=11.5,
    )

    png = (
        PLOT_DIR
        / "03_xgboost_training_holdout_parity.png"
    )

    pdf = (
        PLOT_DIR
        / "03_xgboost_training_holdout_parity.pdf"
    )

    svg = (
        PLOT_DIR
        / "03_xgboost_training_holdout_parity.svg"
    )

    fig.savefig(
        png,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        svg,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    pd.DataFrame([
        {
            "subset": "Training",
            **train_metrics,
        },
        {
            "subset": "Holdout test",
            **holdout_metrics,
        },
    ]).to_csv(
        PLOT_DIR
        / "03_xgboost_training_holdout_metrics.csv",
        index=False,
    )


# =============================================================================
# TAYLOR STATISTICS
# =============================================================================

def calculate_taylor_statistics(
    observed,
    predicted,
):

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    observed_std = np.std(
        observed,
        ddof=1,
    )

    predicted_std = np.std(
        predicted,
        ddof=1,
    )

    correlation = np.corrcoef(
        observed,
        predicted,
    )[0, 1]

    observed_anomaly = (
        observed
        -
        np.mean(observed)
    )

    predicted_anomaly = (
        predicted
        -
        np.mean(predicted)
    )

    centered_rmse = np.sqrt(
        np.mean(
            (
                predicted_anomaly
                -
                observed_anomaly
            ) ** 2
        )
    )

    return {
        "correlation": float(correlation),
        "observed_std": float(observed_std),
        "predicted_std": float(predicted_std),
        "normalized_std": float(
            predicted_std
            /
            observed_std
        ),
        "centered_rmse": float(
            centered_rmse
        ),
        "normalized_centered_rmse": float(
            centered_rmse
            /
            observed_std
        ),
    }


# =============================================================================
# TAYLOR CORRELATION-AXIS TITLE
# =============================================================================

def add_correlation_axis_title(
    ax,
    rmax,
):

    # Label positions and rotations follow the outer arc.

    radius = (
        rmax
        *
        1.085
    )

    # First word.
    theta1_deg = 47.0

    ax.text(
        np.deg2rad(theta1_deg),
        radius,
        "Correlation",
        fontsize=13.0,
        fontweight="bold",
        ha="center",
        va="center",
        rotation=(
            theta1_deg
            -
            90.0
        ),
        rotation_mode="anchor",
        clip_on=False,
        zorder=30,
    )

    # Second word moved substantially closer to the first.
    theta2_deg = 34.0

    ax.text(
        np.deg2rad(theta2_deg),
        radius,
        "coefficient",
        fontsize=13.0,
        fontweight="bold",
        ha="center",
        va="center",
        rotation=(
            theta2_deg
            -
            90.0
        ),
        rotation_mode="anchor",
        clip_on=False,
        zorder=30,
    )


# =============================================================================
# TAYLOR DIAGRAM
# =============================================================================

def make_taylor_diagram(loaded):

    reference = loaded[
        "pls"
    ][
        "holdout"
    ][
        "observed_chl_a"
    ].to_numpy(
        dtype=float
    )

    rows = []

    for key, name in MODELS:

        holdout = loaded[
            key
        ][
            "holdout"
        ]

        observed = holdout[
            "observed_chl_a"
        ].to_numpy(
            dtype=float
        )

        predicted = holdout[
            "predicted_chl_a"
        ].to_numpy(
            dtype=float
        )

        if len(observed) != len(reference):

            raise RuntimeError(
                f"{name} has a different holdout size."
            )

        if not np.allclose(
            observed,
            reference,
        ):

            raise RuntimeError(
                f"{name} does not use the same "
                f"ordered holdout observations."
            )

        stats = calculate_taylor_statistics(
            observed,
            predicted,
        )

        rows.append({
            "model": name,
            **stats,
        })

    taylor_df = pd.DataFrame(
        rows
    )

    taylor_df.to_csv(
        PLOT_DIR
        / "02_taylor_statistics.csv",
        index=False,
    )


    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------

    fig = plt.figure(
        figsize=(
            9.5,
            9.0,
        )
    )

    ax = fig.add_subplot(
        111,
        projection="polar",
    )

    ax.set_axisbelow(True)

    ax.set_theta_zero_location(
        "E"
    )

    ax.set_theta_direction(
        1
    )

    ax.set_thetamin(
        0
    )

    ax.set_thetamax(
        90
    )


    # -------------------------------------------------------------------------
    # Correlation scale
    # -------------------------------------------------------------------------

    correlation_ticks = np.array([
        0.0,
        0.2,
        0.4,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.99,
        1.0,
    ])

    _, correlation_labels = ax.set_thetagrids(
        np.degrees(
            np.arccos(
                correlation_ticks
            )
        ),
        labels=[
            f"{value:g}"
            for value in correlation_ticks
        ],
    )

    for label in correlation_labels:

        label.set_fontsize(
            12.5
        )


    # -------------------------------------------------------------------------
    # Radial extent
    # -------------------------------------------------------------------------

    maximum_model_std = float(
        taylor_df[
            "normalized_std"
        ].max()
    )

    rmax = max(
        1.15,
        maximum_model_std
        *
        1.18,
    )

    ax.set_ylim(
        0,
        rmax,
    )


    # -------------------------------------------------------------------------
    # Radial ticks
    # -------------------------------------------------------------------------

    radial_ticks = np.arange(
        0.2,
        1.01,
        0.2,
    )

    ax.set_yticks(
        radial_ticks
    )

    ax.set_yticklabels(
        [
            f"{value:.1f}"
            for value in radial_ticks
        ],
        fontsize=10,
        color="0.45",
    )

    ax.set_rlabel_position(
        112
    )


    # -------------------------------------------------------------------------
    # Standard deviation label close to actual axes.
    # -------------------------------------------------------------------------

    ax.text(
        -0.045,
        0.50,
        "Normalized standard deviation",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        clip_on=False,
    )


    # -------------------------------------------------------------------------
    # Correlation title
    # -------------------------------------------------------------------------

    add_correlation_axis_title(
        ax,
        rmax,
    )


    # -------------------------------------------------------------------------
    # Observed standard-deviation arc
    # -------------------------------------------------------------------------

    theta = np.linspace(
        0,
        np.pi / 2,
        400,
    )

    ax.plot(
        theta,
        np.ones_like(theta),
        linestyle="--",
        color="0.35",
        linewidth=1.45,
        alpha=0.80,
        zorder=2,
    )


    # -------------------------------------------------------------------------
    # Normalized centered-RMSE contours
    # -------------------------------------------------------------------------

    radial_grid = np.linspace(
        0,
        rmax,
        320,
    )

    theta_grid = np.linspace(
        0,
        np.pi / 2,
        320,
    )

    T, R = np.meshgrid(
        theta_grid,
        radial_grid,
    )

    centered_error = np.sqrt(
        1.0
        +
        R ** 2
        -
        2.0
        *
        R
        *
        np.cos(T)
    )

    contour_levels = np.array([
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
    ])

    contour_levels = contour_levels[
        contour_levels
        <
        np.nanmax(centered_error)
    ]

    contours = ax.contour(
        T,
        R,
        centered_error,
        levels=contour_levels,
        colors="0.55",
        linestyles="dotted",
        linewidths=1.0,
        zorder=1,
    )

    ax.clabel(
        contours,
        inline=True,
        fontsize=9.5,
        fmt="%.1f",
    )


    # -------------------------------------------------------------------------
    # MODEL COLORS
    #
    # Make PLS dark blue and Kernel PLS bright cyan for clear separation.
    # -------------------------------------------------------------------------

    model_colors = {

        "PLS":
            "#24527A",      # deep blue

        "Kernel PLS":
            "#00B8D4",      # bright cyan

        "MLP":
            "#2ca02c",      # green

        "RealMLP":
            "#d62728",      # red

        "Random Forest":
            "#9467bd",      # purple

        "XGBoost":
            "#f2a900",      # gold/orange

    }


    handles = []
    labels = []


    for _, row in taylor_df.iterrows():

        model_name = row[
            "model"
        ]

        correlation = np.clip(
            float(
                row[
                    "correlation"
                ]
            ),
            0.0,
            1.0,
        )

        angle = np.arccos(
            correlation
        )

        radius = float(
            row[
                "normalized_std"
            ]
        )


        # SAME marker size for all six models.
        marker_size = 155


        if model_name == "XGBoost":

            edge_color = "#b22222"
            line_width = 2.3
            zorder = 9

        else:

            edge_color = "black"
            line_width = 1.25
            zorder = 7


        point = ax.scatter(
            angle,
            radius,
            s=marker_size,
            marker="o",
            facecolor=model_colors[
                model_name
            ],
            edgecolor=edge_color,
            linewidth=line_width,
            zorder=zorder,
        )

        handles.append(
            point
        )

        labels.append(
            model_name
        )


    # -------------------------------------------------------------------------
    # Observed reference
    # -------------------------------------------------------------------------

    observed_reference = ax.scatter(
        0,
        1.0,
        marker="*",
        s=285,
        facecolor="white",
        edgecolor="black",
        linewidth=1.8,
        zorder=12,
        clip_on=False,
    )

    handles.append(
        observed_reference
    )

    labels.append(
        "Observed reference"
    )


    # -------------------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------------------

    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.075,
        ),
        ncol=4,
        frameon=True,
        framealpha=0.97,
        fontsize=12,
        borderpad=0.70,
        columnspacing=1.20,
        labelspacing=0.50,
        handletextpad=0.55,
    )


    ax.tick_params(
        axis="both",
        which="major",
        labelsize=11.5,
        pad=8,
    )

    ax.grid(
        True,
        alpha=0.27,
    )


    fig.subplots_adjust(
        left=0.060,
        right=0.97,
        top=0.87,
        bottom=0.18,
    )


    png = (
        PLOT_DIR
        / "02_taylor_diagram_six_models.png"
    )

    pdf = (
        PLOT_DIR
        / "02_taylor_diagram_six_models.pdf"
    )

    svg = (
        PLOT_DIR
        / "02_taylor_diagram_six_models.svg"
    )


    fig.savefig(
        png,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        svg,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    return taylor_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 100)
    print("LOADING SAVED SIX-MODEL PREDICTIONS")
    print("=" * 100)

    loaded = load_predictions()


    # -------------------------------------------------------------------------
    # Six-model holdout
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("CREATING SIX-MODEL HOLDOUT FIGURE")
    print("=" * 100)

    comparison_metrics = make_six_model_figure(
        loaded
    )

    print(
        comparison_metrics[
            [
                "model",
                "r2",
                "rmse",
                "mae",
                "bias",
                "slope",
                "intercept",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


    # -------------------------------------------------------------------------
    # Taylor
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("CREATING TAYLOR DIAGRAM")
    print("=" * 100)

    taylor_df = make_taylor_diagram(
        loaded
    )

    print(
        taylor_df[
            [
                "model",
                "correlation",
                "normalized_std",
                "normalized_centered_rmse",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


    # -------------------------------------------------------------------------
    # XGBoost training + holdout
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("CREATING XGBOOST TRAINING + HOLDOUT FIGURE")
    print("=" * 100)

    make_xgboost_training_holdout(
        loaded
    )


    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print(
        f"Final figures:\n{PLOT_DIR}"
    )

    print()
    print("Main outputs:")
    print("  01_six_model_holdout_parity.png/.pdf/.svg")
    print("  02_taylor_diagram_six_models.png/.pdf/.svg")
    print("  03_xgboost_training_holdout_parity.png/.pdf/.svg")

    print()
    print(
        "Parity axes: 0.05 to 300 µg/L; "
        "0.05 is not labeled."
    )

    print(
        "Red lines and equations are ordinary least-squares fits "
        "calculated from original Chl-a concentrations."
    )

    print(
        "Logarithmic axes are used only for visualization."
    )

