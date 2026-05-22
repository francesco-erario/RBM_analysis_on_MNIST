"""
Plotting utilities for RBM training results.

Every function takes the *results* dict produced by ``RBM.train()``
(or loaded via ``load_results()``) and returns a matplotlib ``Figure``.

Typical workflow
----------------
>>> from rbm import RBM, load_results
>>> results = load_results("run_01")          # or model.train(...)
>>>
>>> from rbm_plots import plot_metrics, plot_filters
>>> fig = plot_metrics(results)
>>> fig.savefig("metrics.png")
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm


def _close_and_return(fig):
    """Prevent double-render in notebooks."""
    # plt.close(fig)
    return fig


# =========================================================================
#  1.  TRAINING METRICS  (4 panels)
# =========================================================================

def plot_metrics(results: dict) -> plt.Figure:
    """
    Four-panel overview of training quality:

    1. Free energy — data vs fantasy
    2. Free energy gap
    3. Reconstruction error
    4. Pseudo-likelihood

    Requires ``metrics_every > 0`` during training.
    """
    m = results["metrics"]
    if len(m["epochs"]) == 0:
        raise ValueError("No metrics recorded (metrics_every was 0?).")

    ep   = m["epochs"]
    fe_d = m["free_energy_data"]
    fe_f = m["free_energy_fantasy"]
    gap  = m["free_energy_gap"]
    re   = m["reconstruction_error"]
    pl   = m["pseudo_likelihood"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    ax.plot(ep, fe_d, label="F(data)",    color="#2266CC", lw=1.5)
    ax.plot(ep, fe_f, label="F(fantasy)", color="#CC4422", lw=1.5, ls="--")
    ax.set(xlabel="Epoch", ylabel="Mean free energy",
           title="Free energy: data vs fantasy")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ep, gap, color="#884488", lw=1.5)
    ax.fill_between(ep, 0, gap, alpha=0.15, color="#884488")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set(xlabel="Epoch", ylabel="F(data) − F(fantasy)",
           title="Free-energy gap (→ 0)")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ep, re, color="#228833", lw=1.5)
    ax.set(xlabel="Epoch", ylabel="MSE",
           title="Reconstruction error (↓)")
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(ep, pl, color="#CC8822", lw=1.5)
    ax.set(xlabel="Epoch", ylabel="log PL",
           title="Pseudo-likelihood (↑)")
    ax.grid(alpha=0.3)

    L = results["config"]["L"]
    fig.suptitle(f"Training metrics — L = {L}", fontsize=14)
    return _close_and_return(fig)


# =========================================================================
#  2.  CD ANALYSIS  (3 panels)
# =========================================================================

def plot_cd_analysis(results: dict) -> plt.Figure:
    """
    Three-panel Contrastive-Divergence decomposition:

    1. Std of positive vs negative phase (log-log)
    2. Relative gap between the two phases
    3. Gradient distribution at early / mid / late epochs
    """
    h = results["history"]
    if len(h["epochs"]) == 0:
        raise ValueError("No history recorded (history_every was 0?).")

    epochs   = h["epochs"]
    gw_data  = h["gw_data"]
    gw_model = h["gw_model"]
    gw       = h["gw"]
    n = len(epochs)

    std_d = np.array([np.std(gw_data[i])  for i in range(n)])
    std_m = np.array([np.std(gw_model[i]) for i in range(n)])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    # Panel 1 — positive vs negative phase
    ax = axes[0]
    ax.plot(epochs, std_d, label=r"$\langle xz\rangle_\mathrm{data}$",
            color="#2266CC", lw=1.5)
    ax.plot(epochs, std_m, label=r"$\langle xz\rangle_\mathrm{model}$",
            color="#CC4422", lw=1.5)
    ax.set(xlabel="Epoch", ylabel="Std of gradient term",
           title="CD: positive vs negative phase", xscale="log", yscale="log")
    ax.legend(); ax.grid(alpha=0.3)

    # Panel 2 — relative gap
    ax = axes[1]
    rel = np.abs(std_d - std_m) / (std_d + 1e-10)
    ax.plot(epochs, rel, color="#884488", lw=1.5)
    ax.fill_between(epochs, 0, rel, alpha=0.2, color="#884488")
    ax.set(xlabel="Epoch", ylabel="|data−model| / data",
           title="Relative gap", xscale="log")
    ax.grid(alpha=0.3)

    # Panel 3 — gradient distribution snapshots
    ax = axes[2]
    mid = n // 2
    for idx, col, lab in [
        (0,     "#AABBFF", f"ep {epochs[0]}"),
        (mid,   "#5577EE", f"ep {epochs[mid]}"),
        (n - 1, "#1133AA", f"ep {epochs[-1]}"),
    ]:
        ax.hist(gw[idx].ravel(), bins=60, density=True,
                alpha=0.5, color=col, label=lab)
    ax.set(xlabel="Gradient value", ylabel="Density",
           title=r"Distribution of $\nabla_w$")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("CD analysis", fontsize=14)
    return _close_and_return(fig)


# =========================================================================
#  3.  LEARNED FILTERS (single weight matrix)
# =========================================================================

def plot_filters(
    w: np.ndarray,
    side: int,
    *,
    ncols: int = 6,
    vmax: float = 4.0,
    title: str = "Learned filters",
    title_color: str = "black",
) -> plt.Figure:
    """
    Display every column of *w* as a ``side × side`` image.

    Parameters
    ----------
    w : ndarray, shape (D, L)
        Weight matrix (e.g. ``results["weights"]["w"]``).
    side : int
        Image side length (``side² == D``).
    """
    L = w.shape[1]
    nrows = int(np.ceil(L / ncols))

    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(2 * ncols, 2 * nrows),
                            constrained_layout=True)
    axs = np.atleast_2d(axs)

    for j in range(L):
        ax = axs[j // ncols, j % ncols]
        ax.imshow(w[:, j].reshape(side, side), cmap="bwr",
                  vmin=-vmax, vmax=vmax)
        ax.set(xticks=[], yticks=[])
        ax.set_title(f"z{j+1}", fontsize=8)

    for j in range(L, nrows * ncols):
        axs[j // ncols, j % ncols].axis("off")

    fig.suptitle(title, color=title_color, fontsize=14)
    return _close_and_return(fig)


# =========================================================================
#  4.  WEIGHT EVOLUTION  (snapshot grid)
# =========================================================================

def plot_weight_evolution(
    results: dict,
    side: int,
    *,
    checkpoints: list[int] | None = None,
    vmax: float = 4.0,
) -> plt.Figure:
    """
    Grid showing filters and visible bias at selected epochs.

    Parameters
    ----------
    side : int
        Image side.
    checkpoints : list[int] | None
        Epoch indices to show (taken from history).
        *None* → automatic spread.
    """
    h = results["history"]
    hist_ep = h["epochs"]
    wE = h["w"]
    aE = h["a"]
    n = len(hist_ep)
    L = results["config"]["L"]

    if checkpoints is None:
        pick = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        pick = sorted(set(np.clip(pick, 0, n - 1)))
    else:
        # map requested epochs → nearest available history index
        pick = [int(np.argmin(np.abs(hist_ep - ep))) for ep in checkpoints]
        pick = sorted(set(pick))

    fig, axes = plt.subplots(
        len(pick), L + 1,
        figsize=(1.6 * (L + 1), 1.8 * len(pick)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for row, idx in enumerate(pick):
        # visible bias
        ax = axes[row, 0]
        ax.imshow(aE[idx].reshape(side, side), cmap="bwr",
                  vmin=-vmax, vmax=vmax)
        ax.set(xticks=[], yticks=[])
        ax.set_ylabel(f"ep {hist_ep[idx]}", fontsize=9)
        if row == 0:
            ax.set_title("bias a", fontsize=9)

        # hidden filters
        for j in range(L):
            ax = axes[row, j + 1]
            ax.imshow(wE[idx][:, j].reshape(side, side), cmap="bwr",
                      vmin=-vmax, vmax=vmax)
            ax.set(xticks=[], yticks=[])
            if row == 0:
                ax.set_title(f"z{j+1}", fontsize=8)

    fig.suptitle(f"Weight evolution — L = {L}", fontsize=14)
    return _close_and_return(fig)


# =========================================================================
#  5.  GENERATIVE CHAIN (digit classification over Gibbs steps)
# =========================================================================

def plot_chain_digits(
    digit_sequences: list[list[int]],
    *,
    start_labels: list[int] | None = None,
    n_classes: int = 10,
    title: str = "Digit classification along Gibbs chain",
) -> plt.Figure:
    """
    Plot digit-class trajectories from Gibbs chains.

    Parameters
    ----------
    digit_sequences : list of lists
        Each inner list is the classified digit at every chain step.
        Generate these externally with ``RBM.sample_chain`` +
        ``classify_nearest``.
    start_labels : list[int] | None
        Starting digit for each chain (for labelling).
    """
    n_chains = len(digit_sequences)
    fig, axes = plt.subplots(n_chains, 1,
                             figsize=(8, 1.8 * n_chains),
                             squeeze=False,
                             constrained_layout=True)

    for i, seq in enumerate(digit_sequences):
        ax = axes[i, 0]
        ax.plot(seq, lw=0.8, color=f"C{i % 10}")
        ax.set_ylim(-0.3, n_classes - 0.7)
        ax.set_yticks(range(n_classes))
        ax.grid(alpha=0.3)
        if i == n_chains - 1:
            ax.set_xlabel("Gibbs step")
        lbl = start_labels[i] if start_labels else i
        ax.set_ylabel(f"start={lbl}")

    fig.suptitle(title, fontsize=14)
    return _close_and_return(fig)


# =========================================================================
#  6.  COMPARE MULTIPLE RUNS  (e.g. different L values)
# =========================================================================

def plot_convergence_comparison(
    runs: dict[str, dict],
    *,
    metric: str = "gradient_rms",
    title: str = "Convergence comparison",
) -> plt.Figure:
    """
    Overlay a scalar curve from several training runs.

    Parameters
    ----------
    runs : dict
        ``{label: results_dict}``.  Labels are used in the legend.
    metric : str
        ``"gradient_rms"``  — RMS of weight gradient (from history).
        ``"reconstruction_error"`` | ``"pseudo_likelihood"``
        ``"free_energy_gap"`` — from metrics.
    """
    cmap = mpl_cm.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    for i, (label, res) in enumerate(runs.items()):
        color = cmap(i % 10)

        if metric == "gradient_rms":
            h  = res["history"]
            ep = h["epochs"]
            vals = np.array([np.std(h["gw"][j]) for j in range(len(ep))])
            ax.set(yscale="log", ylabel="Gradient RMS (w)")
        else:
            m  = res["metrics"]
            ep = m["epochs"]
            vals = m[metric]
            ax.set(ylabel=metric.replace("_", " ").title())

        ax.plot(ep, vals, label=label, color=color, lw=1.5)

    ax.set(xlabel="Epoch", xscale="log", title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    return _close_and_return(fig)
