"""
stochastic.py — Gradient stochastique et sous-gradient stochastique (TP4)
=========================================================================
Contenu :
  1. SGD — régression polynomiale (point unique + mini-lot)
  2. SSGD — classification binaire (point unique + mini-lot)
  3. Comparaison SGD vs gradient complet
  4. Visualisation
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath('..'))
from TP1.polynomial import (phi, predict as poly_predict, mse,
                         gradient_mse, gradient_mse_single,
                         gradient_mse_batch)
from TP2.perceptron import (predict, hinge_loss, perceptron_loss,
                         subgradient_batch, subgradient_hinge_batch,
                         subgradient_hinge_individual)
from utils import project_l2_ball


# ===========================================================================
# UTILITAIRES COMMUNS
# ===========================================================================

def _make_step(eta0, t, decay="constant", beta=0.5):
    """Calcule le pas à l'itération t selon la stratégie choisie."""
    if decay == "constant":
        return eta0
    elif decay == "sqrt":
        return eta0 / np.sqrt(t + 1)
    elif decay == "inv":
        return eta0 / (1 + t)
    elif decay == "poly":
        return eta0 / ((t + 1) ** beta)
    return eta0


def _shuffle_indices(n, rng):
    """Retourne les indices mélangés aléatoirement."""
    return rng.permutation(n)


# ===========================================================================
# 1. SGD — RÉGRESSION POLYNOMIALE
# ===========================================================================

def sgd_regression(X, y, d, eta0=0.01, decay="constant", beta=0.5,
                    batch_size=1, n_epochs=10, theta0=None,
                    shuffle=True, seed=42, project_radius=None):
    """
    Gradient stochastique pour la régression polynomiale.

    Paramètres
    ----------
    batch_size    : int — taille du mini-lot (1 = point unique)
    n_epochs      : int — nombre de passes sur les données
    shuffle       : bool — mélanger à chaque époque
    decay         : "constant" | "sqrt" | "inv" | "poly"

    Retourne
    --------
    theta   : paramètres finaux
    history : dict avec 'cost', 'grad_norm', 'step'
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    theta = np.zeros(d + 1) if theta0 is None else theta0.copy()
    history = {"cost": [], "grad_norm": [], "step": []}
    k = 0  # compteur global d'itérations

    for epoch in range(n_epochs):
        indices = _shuffle_indices(n, rng) if shuffle else np.arange(n)

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_b, y_b = X[batch_idx], y[batch_idx]

            # Gradient sur le mini-lot
            if batch_size == 1:
                g = gradient_mse_single(theta, X_b[0], y_b[0], d)
            else:
                g = gradient_mse_batch(theta, X_b, y_b, d)

            step = _make_step(eta0, k, decay, beta)
            theta = theta - step * g

            if project_radius is not None:
                theta = project_l2_ball(theta, project_radius)

            history["cost"].append(mse(theta, X, y, d))
            history["grad_norm"].append(float(np.linalg.norm(g)))
            history["step"].append(step)
            k += 1

    return theta, history


def compare_sgd_regression(X, y, d, eta0=0.05, n_epochs=5, seed=42):
    """
    Compare plusieurs configurations SGD pour la régression.

    Configurations testées :
      - Point unique, ordre fixe
      - Point unique, ordre aléatoire
      - Mini-lot (32), ordre aléatoire
      - Pas décroissant (sqrt)
    """
    results = {}
    configs = [
        {"batch_size": 1,  "shuffle": False, "decay": "constant",
         "label": "Point unique — ordre fixe"},
        {"batch_size": 1,  "shuffle": True,  "decay": "constant",
         "label": "Point unique — aléatoire"},
        {"batch_size": 32, "shuffle": True,  "decay": "constant",
         "label": "Mini-lot 32 — aléatoire"},
        {"batch_size": 1,  "shuffle": True,  "decay": "sqrt",
         "label": "Point unique — pas √t"},
    ]
    for cfg in configs:
        label = cfg.pop("label")
        theta, hist = sgd_regression(
            X, y, d, eta0=eta0, n_epochs=n_epochs,
            seed=seed, **cfg
        )
        results[label] = (theta, hist)

    return results


# ===========================================================================
# 2. SSGD — CLASSIFICATION BINAIRE
# ===========================================================================

def ssgd_classification(X, y, eta0=0.01, decay="constant", beta=0.5,
                          batch_size=1, n_epochs=10, w0=None, b0=0.0,
                          shuffle=True, seed=42, project_radius=None,
                          loss="hinge"):
    """
    Sous-gradient stochastique pour la classification binaire.

    Paramètres
    ----------
    loss : "hinge" | "perceptron"

    Retourne
    --------
    w, b    : paramètres finaux
    history : dict avec 'cost', 'grad_norm', 'accuracy', 'step'
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    p = X.shape[1]
    w = np.zeros(p) if w0 is None else w0.copy()
    b = float(b0)

    loss_fn = hinge_loss if loss == "hinge" else perceptron_loss
    grad_fn = subgradient_hinge_batch if loss == "hinge" else subgradient_batch

    history = {"cost": [], "grad_norm": [], "accuracy": [], "step": []}
    k = 0

    for epoch in range(n_epochs):
        indices = _shuffle_indices(n, rng) if shuffle else np.arange(n)

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_b, y_b = X[batch_idx], y[batch_idx]

            if batch_size == 1:
                gw, gb = subgradient_hinge_individual(w, b, X_b[0], y_b[0])
            else:
                gw, gb = grad_fn(w, b, X_b, y_b)

            step = _make_step(eta0, k, decay, beta)
            w = w - step * gw
            b = b - step * gb

            if project_radius is not None:
                w = project_l2_ball(w, project_radius)

            history["cost"].append(loss_fn(w, b, X, y))
            history["grad_norm"].append(float(np.linalg.norm(np.append(gw, gb))))
            history["accuracy"].append(float(np.mean(predict(w, b, X) == y)))
            history["step"].append(step)
            k += 1

    return w, b, history


def compare_ssgd_classification(X, y, eta0=0.05, n_epochs=5, seed=42):
    """
    Compare plusieurs configurations SSGD pour la classification.
    """
    results = {}
    configs = [
        {"batch_size": 1,  "shuffle": False, "decay": "constant",
         "label": "Point unique — ordre fixe"},
        {"batch_size": 1,  "shuffle": True,  "decay": "constant",
         "label": "Point unique — aléatoire"},
        {"batch_size": 32, "shuffle": True,  "decay": "constant",
         "label": "Mini-lot 32 — aléatoire"},
        {"batch_size": 1,  "shuffle": True,  "decay": "sqrt",
         "label": "Point unique — pas √t"},
    ]
    for cfg in configs:
        label = cfg.pop("label")
        w, b, hist = ssgd_classification(
            X, y, eta0=eta0, n_epochs=n_epochs,
            seed=seed, **cfg
        )
        results[label] = (w, b, hist)

    return results


# ===========================================================================
# 3. COMPARAISON SGD vs GRADIENT COMPLET
# ===========================================================================

def sgd_vs_full_regression(X, y, d, eta=0.005, n_iter=500, seed=42):
    """
    Compare SGD (point unique) et gradient complet sur la régression.

    Retourne
    --------
    hist_sgd  : history SGD
    hist_full : history gradient complet
    """
    from TP1.polynomial import gradient_descent

    # SGD — autant d'itérations que le gradient complet
    theta_sgd, hist_sgd = sgd_regression(
        X, y, d, eta0=eta, decay="constant",
        batch_size=1, n_epochs=1, shuffle=True,
        seed=seed
    )
    # On tronque / complète pour avoir n_iter points
    # (une époque = n exemples = n steps)

    # Gradient complet
    theta_full, hist_full = gradient_descent(
        X, y, d, alpha=eta, n_iter=n_iter
    )

    return hist_sgd, hist_full


def sgd_vs_full_classification(X, y, eta=0.01, n_epochs=5, seed=42):
    """
    Compare SSGD (point unique) et sous-gradient complet sur la classification.
    """
    from TP2.perceptron import subgradient_descent

    # SSGD
    _, _, hist_ssgd = ssgd_classification(
        X, y, eta0=eta, n_epochs=n_epochs,
        batch_size=1, shuffle=True, seed=seed
    )

    # Sous-gradient complet (même nombre d'itérations approx.)
    n_iter_batch = n_epochs * len(y)
    _, _, hist_full = subgradient_descent(
        X, y, alpha=eta, n_iter=n_iter_batch, loss="hinge"
    )

    return hist_ssgd, hist_full


# ===========================================================================
# 4. VISUALISATION
# ===========================================================================

COLORS = ["#378ADD", "#D85A30", "#1D9E75", "#D4537E",
          "#EF9F27", "#7F77DD", "#888780"]


def plot_sgd_comparison(results, metric="cost",
                         title="Comparaison SGD", ylabel=None):
    """
    Trace une métrique pour toutes les configurations SGD/SSGD.

    results : dict {label: (theta_or_w, [b,] history)}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, vals) in enumerate(results.items()):
        hist = vals[-1]   # history est toujours le dernier élément
        if metric not in hist:
            continue
        values = np.array(hist[metric])
        # Lissage léger
        window = max(1, len(values) // 80)
        smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=label, color=COLORS[i % len(COLORS)],
                linewidth=2)
    ax.set_xlabel("Itération (update)")
    ax.set_ylabel(ylabel or metric.capitalize())
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_sgd_vs_full(hist_sgd, hist_full, metric="cost",
                      label_sgd="SGD", label_full="Batch",
                      title="SGD vs Gradient complet"):
    """Compare SGD et gradient complet sur 2 sous-graphes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # SGD : lissé
    sgd_vals = np.array(hist_sgd[metric])
    window = max(1, len(sgd_vals) // 60)
    sgd_smooth = np.convolve(sgd_vals, np.ones(window)/window, mode='valid')
    axes[0].plot(sgd_smooth, color="#378ADD", linewidth=2, label=label_sgd)
    axes[0].set_title(f"{label_sgd} — {metric}")
    axes[0].set_xlabel("Update")
    axes[0].set_ylabel(metric)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    # Batch
    axes[1].plot(hist_full[metric], color="#D85A30", linewidth=2,
                 label=label_full)
    axes[1].set_title(f"{label_full} — {metric}")
    axes[1].set_xlabel("Itération")
    axes[1].set_ylabel(metric)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    plt.suptitle(title, y=1.01)
    plt.tight_layout()
    return fig


def plot_variance(results, metric="cost",
                   title="Variance des mises à jour — SGD"):
    """
    Trace la variance locale des mises à jour pour chaque configuration.
    Fenêtre glissante de 50 steps.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (label, vals) in enumerate(results.items()):
        hist = vals[-1]
        if metric not in hist:
            continue
        values = np.array(hist[metric])
        window = 50
        variances = [np.var(values[max(0, j-window):j+1])
                     for j in range(len(values))]
        ax.plot(variances, label=label, color=COLORS[i % len(COLORS)],
                linewidth=1.5, alpha=0.85)
    ax.set_xlabel("Itération")
    ax.set_ylabel("Variance locale")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_step_evolution(results, title="Évolution du pas ηk"):
    """Trace l'évolution du pas pour chaque configuration."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (label, vals) in enumerate(results.items()):
        hist = vals[-1]
        if "step" not in hist:
            continue
        ax.plot(hist["step"], label=label,
                color=COLORS[i % len(COLORS)], linewidth=2)
    ax.set_xlabel("Itération")
    ax.set_ylabel("ηk")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig