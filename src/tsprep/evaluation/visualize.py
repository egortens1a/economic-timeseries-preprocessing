"""Графики обучения, реконструкции и латентных представлений."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_loss_curves(history, title: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.train_loss, label="train")
    axes[0].plot(history.val_loss, label="val")
    axes[0].set_title(f"Loss{': ' + title if title else ''}")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.val_recon, label="val recon")
    axes[1].plot(history.val_kl, label="val KL")
    axes[1].plot(history.beta, label="beta", linestyle="--")
    axes[1].set_title("Val recon / KL / beta(annealing)")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_reconstruction_examples(x: np.ndarray, x_hat: np.ndarray, n_examples: int = 4, feature: int = 0):
    idx = np.linspace(0, x.shape[0] - 1, n_examples).astype(int)
    fig, axes = plt.subplots(n_examples, 1, figsize=(9, 2.2 * n_examples), sharex=True)
    if n_examples == 1:
        axes = [axes]
    for ax, i in zip(axes, idx):
        ax.plot(x[i, :, feature], label="original")
        ax.plot(x_hat[i, :, feature], label="reconstruction", linestyle="--")
        ax.set_ylabel(f"win #{i}")
        ax.set_xticks(range(0, x.shape[1], max(1, x.shape[1] // 8)))
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Примеры реконструкции окон (нормализованный масштаб)")
    fig.tight_layout()
    return fig


def plot_latent_projection(proj: np.ndarray, labels: np.ndarray | None = None, title: str = ""):
    fig, ax = plt.subplots(figsize=(6, 5))
    if labels is not None:
        scatter = ax.scatter(proj[:, 0], proj[:, 1], c=labels, s=8, alpha=0.7)
        legend = ax.legend(*scatter.legend_elements(), title="regime", loc="best", fontsize=8)
        ax.add_artist(legend)
    else:
        ax.scatter(proj[:, 0], proj[:, 1], s=8, alpha=0.7)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def comparison_table(results: dict) -> pd.DataFrame:
    """Собирает метрики моделей в таблицу, отсортированную по MSE."""
    df = pd.DataFrame(results).T
    if "mse" in df.columns:
        df = df.sort_values("mse")
    return df


def plot_task_metric(frame, metric, title=None):
    """Compare methods within dataset/fold; error bars describe initialization SD."""
    if frame.empty or metric not in frame:
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.text(.5, .5, "No measurements for this task", ha="center", va="center")
        ax.set_axis_off()
        return fig
    keys = [key for key in ("dataset", "fold", "unit", "label_source") if key in frame]
    groups = list(frame.groupby(keys, dropna=False))
    columns = min(3, len(groups))
    rows = (len(groups)+columns-1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(7*columns, 5*rows), squeeze=False)
    for axis, (group, values) in zip(axes.flat, groups):
        summary = values.groupby("model")[metric].agg(["mean", "std", "count"]).dropna(subset=["mean"])
        if len(summary):
            axis.barh(summary.index, summary["mean"], xerr=summary["std"].fillna(0))
            axis.axvline(0, color="black", linewidth=.7)
        else:
            axis.text(.5, .5, "Metric is undefined for this split", ha="center", va="center", transform=axis.transAxes)
        group = group if isinstance(group, tuple) else (group,)
        axis.set_title(", ".join(f"{key}={value}" for key, value in zip(keys, group)))
        axis.set_xlabel(metric)
        axis.grid(axis="x", alpha=.2)
    for axis in list(axes.flat)[len(groups):]:
        axis.set_axis_off()
    fig.suptitle((title or metric)+"; mean ± initialization SD (not CI)")
    fig.tight_layout()
    return fig
