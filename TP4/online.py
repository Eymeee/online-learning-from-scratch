"""
online.py — Gradient et sous-gradient en ligne (TP4)
=====================================================
Contenu :
  1. Gradient en ligne projeté — régression polynomiale
  2. Sous-gradient en ligne projeté — classification binaire
  3. Calcul du regret
  4. Visualisation
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath('..'))
from TP1.polynomial import (
    phi,
    predict as poly_predict,
    mse,
    gradient_mse_single,
    gradient_descent,
)
from TP2.perceptron import (
    predict,
    hinge_loss,
    subgradient_hinge_individual,
    subgradient_descent,
)
from utils import project_l2_ball


# ===========================================================================
# 1. GRADIENT EN LIGNE — RÉGRESSION POLYNOMIALE
# ===========================================================================

def online_gradient_regression(X, y, d, eta=None, eta0=0.1,
                                 decay="constant", beta=0.5,
                                 theta0=None, project_radius=None):
    """
    Gradient en ligne projeté pour la régression polynomiale.

    À chaque tour t :
      - Reçoit (xt, yt)
      - Calcule la perte instantanée ℓt(theta) = (ŷ(xt) - yt)²
      - Calcule gt = ∇ℓt(theta_t)
      - Met à jour theta_{t+1} = ΠS(theta_t - eta_t * gt)

    Paramètres
    ----------
    X, y          : données (parcourues dans l'ordre)
    d             : degré polynomial
    eta           : pas constant (si fourni, ignore eta0/decay)
    eta0          : pas initial
    decay         : "constant" | "sqrt" (eta0/√t) | "poly" (eta0/(t^beta))
    beta          : exposant pour decay="poly"
    project_radius: rayon de projection L2 (None = pas de projection)

    Retourne
    --------
    theta       : paramètres finaux
    history     : dict avec 'instant_loss', 'cumul_loss', 'theta_list'
    """
    n = len(y)
    theta = np.zeros(d + 1) if theta0 is None else theta0.copy()

    history = {
        "instant_loss": [],
        "cumul_loss":   [],
        "grad_norm":    [],
        "theta_list":   [],
    }
    cumul = 0.0

    for t in range(n):
        # Pas d'apprentissage
        if eta is not None:
            step = eta
        elif decay == "sqrt":
            step = eta0 / np.sqrt(t + 1)
        elif decay == "poly":
            step = eta0 / ((t + 1) ** beta)
        else:
            step = eta0

        # Perte instantanée AVANT mise à jour
        y_hat = phi(X[t], d) @ theta
        loss_t = (y_hat - y[t]) ** 2
        cumul += loss_t

        # Gradient instantané
        g = gradient_mse_single(theta, X[t], y[t], d)

        # Mise à jour
        theta = theta - step * g

        # Projection
        if project_radius is not None:
            theta = project_l2_ball(theta, project_radius)

        history["instant_loss"].append(float(loss_t))
        history["cumul_loss"].append(float(cumul))
        history["grad_norm"].append(float(np.linalg.norm(g)))
        history["theta_list"].append(theta.copy())

    return theta, history


def compare_steps_online_regression(X, y, d, n_iter=None,
                                     eta0=0.1, project_radius=None):
    """
    Compare les 3 suites de pas pour le gradient en ligne — régression.

    Retourne
    --------
    results : dict {nom: history}
    """
    n = n_iter or len(y)
    X_sub, y_sub = X[:n], y[:n]
    results = {}

    for decay, label in [
        ("constant", f"Constant η={eta0}"),
        ("sqrt",     f"Décroissant η₀/√t"),
        ("poly",     f"Décroissant η₀/t^0.5"),
    ]:
        theta0 = np.zeros(d + 1)
        _, hist = online_gradient_regression(
            X_sub, y_sub, d, eta0=eta0, decay=decay,
            theta0=theta0, project_radius=project_radius
        )
        results[label] = hist

    return results


# ===========================================================================
# 2. SOUS-GRADIENT EN LIGNE — CLASSIFICATION BINAIRE
# ===========================================================================

def online_subgradient_classification(X, y, eta=None, eta0=0.1,
                                       decay="constant", beta=0.5,
                                       w0=None, b0=0.0,
                                       project_radius=None):
    """
    Sous-gradient en ligne projeté pour la classification binaire (hinge).

    À chaque tour t :
      - Reçoit (xt, yt)
      - Calcule la perte hinge instantanée ℓt = max(0, 1 - yt*(w^T xt + b))
      - Calcule gt ∈ ∂ℓt(wt, bt)
      - Met à jour (w,b)_{t+1} = ΠS((w,b)_t - eta_t * gt)

    Retourne
    --------
    w, b    : paramètres finaux
    history : dict avec 'instant_loss', 'cumul_loss', 'errors', 'regret_terms'
    """
    n = len(y)
    p = X.shape[1]
    w = np.zeros(p) if w0 is None else w0.copy()
    b = float(b0)

    history = {
        "instant_loss":  [],
        "cumul_loss":    [],
        "grad_norm":     [],
        "errors":        [],      # 1 si erreur de classification
        "cumul_errors":  [],
    }
    cumul = 0.0
    cumul_errors = 0

    for t in range(n):
        # Pas
        if eta is not None:
            step = eta
        elif decay == "sqrt":
            step = eta0 / np.sqrt(t + 1)
        elif decay == "poly":
            step = eta0 / ((t + 1) ** beta)
        else:
            step = eta0

        # Perte hinge instantanée AVANT mise à jour
        margin = y[t] * (np.dot(w, X[t]) + b)
        loss_t = max(0.0, 1.0 - margin)
        cumul += loss_t

        # Erreur de classification
        err = int(margin < 0)
        cumul_errors += err

        # Sous-gradient
        gw, gb = subgradient_hinge_individual(w, b, X[t], y[t])

        # Mise à jour
        w = w - step * gw
        b = b - step * gb

        # Projection
        if project_radius is not None:
            w = project_l2_ball(w, project_radius)

        history["instant_loss"].append(float(loss_t))
        history["cumul_loss"].append(float(cumul))
        history["grad_norm"].append(float(np.linalg.norm(np.append(gw, gb))))
        history["errors"].append(err)
        history["cumul_errors"].append(cumul_errors)

    return w, b, history


def compare_steps_online_classification(X, y, n_iter=None,
                                         eta0=0.1, project_radius=None):
    """
    Compare les 3 suites de pas pour le sous-gradient en ligne — classification.
    """
    n = n_iter or len(y)
    X_sub, y_sub = X[:n], y[:n]
    results = {}

    for decay, label in [
        ("constant", f"Constant η={eta0}"),
        ("sqrt",     f"Décroissant η₀/√t"),
        ("poly",     f"Décroissant η₀/t^0.5"),
    ]:
        _, _, hist = online_subgradient_classification(
            X_sub, y_sub, eta0=eta0, decay=decay,
            project_radius=project_radius
        )
        results[label] = hist

    return results


# ===========================================================================
# 3. REGRET
# ===========================================================================

def estimate_regret(cumul_losses, best_fixed_loss_per_round):
    """
    Regret_T = sum_{t=1}^T ℓt(wt) - sum_{t=1}^T ℓt(w*)

    cumul_losses            : list — pertes cumulées de l'algorithme
    best_fixed_loss_per_round : float — perte moyenne du meilleur prédicteur fixe

    Retourne
    --------
    regret : np.ndarray — regret cumulé à chaque tour
    """
    T = len(cumul_losses)
    algo_cumul = np.array(cumul_losses)
    best_cumul = np.arange(1, T + 1) * best_fixed_loss_per_round
    return algo_cumul - best_cumul


def compute_best_fixed_regression(X, y, d, n_iter=3000, eta=0.005):
    """
    Entraîne un prédicteur batch pour obtenir la perte minimale de référence.
    """
    theta_star, _ = gradient_descent(X, y, d, alpha=eta, n_iter=n_iter)
    return float(mse(theta_star, X, y, d))


def compute_best_fixed_classification(X, y, alpha=0.005, n_iter=3000):
    """
    Entraîne un classifieur batch pour obtenir la perte hinge minimale.
    """
    w_star, b_star, _ = subgradient_descent(
        X, y, alpha=alpha, n_iter=n_iter, loss='hinge'
    )
    return float(hinge_loss(w_star, b_star, X, y))


# ===========================================================================
# 4. VISUALISATION
# ===========================================================================

COLORS = ["#378ADD", "#D85A30", "#1D9E75", "#D4537E", "#EF9F27"]


def plot_instant_losses(results, title="Pertes instantanées en ligne"):
    """Trace les pertes instantanées pour plusieurs configurations."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (label, hist) in enumerate(results.items()):
        losses = np.array(hist["instant_loss"])
        # Lissage pour la lisibilité
        window = max(1, len(losses) // 100)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=label, color=COLORS[i % len(COLORS)],
                linewidth=2)
    ax.set_xlabel("Tour t")
    ax.set_ylabel("Perte instantanée ℓt")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_cumulative_losses(results, title="Perte cumulée en ligne"):
    """Trace les pertes cumulées."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (label, hist) in enumerate(results.items()):
        ax.plot(hist["cumul_loss"], label=label,
                color=COLORS[i % len(COLORS)], linewidth=2)
    ax.set_xlabel("Tour t")
    ax.set_ylabel("Perte cumulée")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_regrets(regrets_dict, title="Regret cumulé"):
    """Trace le regret cumulé pour plusieurs algorithmes."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (label, regret) in enumerate(regrets_dict.items()):
        ax.plot(regret, label=label,
                color=COLORS[i % len(COLORS)], linewidth=2)
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Tour t")
    ax.set_ylabel("Regret cumulé")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_cumul_errors(results, title="Erreurs cumulées — classification en ligne"):
    """Trace le nombre d'erreurs de classification cumulées."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (label, hist) in enumerate(results.items()):
        if "cumul_errors" in hist:
            ax.plot(hist["cumul_errors"], label=label,
                    color=COLORS[i % len(COLORS)], linewidth=2)
    ax.set_xlabel("Tour t")
    ax.set_ylabel("Erreurs cumulées")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_online_vs_batch(online_hist, batch_costs, label_online="En ligne",
                          label_batch="Batch", title="Online vs Batch"):
    """Compare la convergence online et batch."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(online_hist["instant_loss"], alpha=0.4,
                 color="#378ADD", linewidth=1, label="Instantané")
    # Moyenne mobile
    window = max(1, len(online_hist["instant_loss"]) // 50)
    smoothed = np.convolve(online_hist["instant_loss"],
                            np.ones(window)/window, mode='valid')
    axes[0].plot(smoothed, color="#378ADD", linewidth=2,
                 label=f"{label_online} (lissé)")
    axes[0].set_title(f"Perte instantanée — {label_online}")
    axes[0].set_xlabel("Tour t")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(batch_costs, color="#D85A30", linewidth=2,
                 label=label_batch)
    axes[1].set_title(f"Perte globale — {label_batch}")
    axes[1].set_xlabel("Itération")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    plt.suptitle(title, y=1.01)
    plt.tight_layout()
    return fig