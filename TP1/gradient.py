"""
gradient.py — Gradient numérique pour la régression polynomiale (TP1)
======================================================================
Contenu :
  1. Gradient numérique : droite, gauche, centré
  2. Comparaison gradient numérique vs analytique
  3. Étude de la stabilité selon h
  4. Descente de gradient avec gradient numérique + line search
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath('..'))
from polynomial import mse, gradient_mse, phi, predict
from utils import armijo, goldstein, wolfe, project_l2_ball


# ===========================================================================
# 1. GRADIENT NUMÉRIQUE (3 schémas)
# ===========================================================================

def numerical_gradient_forward(f, theta, h=1e-5):
    """
    Schéma différences finies à droite (forward).

    g_k ≈ (f(theta + h*e_k) - f(theta)) / h

    Paramètres
    ----------
    f     : callable(theta) → float
    theta : np.ndarray shape (d+1,)
    h     : float — pas de différentiation

    Retourne
    --------
    grad : np.ndarray shape (d+1,)
    """
    grad = np.zeros_like(theta, dtype=float)
    f0 = f(theta)
    for k in range(len(theta)):
        e_k = np.zeros_like(theta)
        e_k[k] = 1.0
        grad[k] = (f(theta + h * e_k) - f0) / h
    return grad


def numerical_gradient_backward(f, theta, h=1e-5):
    """
    Schéma différences finies à gauche (backward).

    g_k ≈ (f(theta) - f(theta - h*e_k)) / h
    """
    grad = np.zeros_like(theta, dtype=float)
    f0 = f(theta)
    for k in range(len(theta)):
        e_k = np.zeros_like(theta)
        e_k[k] = 1.0
        grad[k] = (f0 - f(theta - h * e_k)) / h
    return grad


def numerical_gradient_centered(f, theta, h=1e-5):
    """
    Schéma différences finies centré (plus précis, ordre 2).

    g_k ≈ (f(theta + h*e_k) - f(theta - h*e_k)) / (2h)
    """
    grad = np.zeros_like(theta, dtype=float)
    for k in range(len(theta)):
        e_k = np.zeros_like(theta)
        e_k[k] = 1.0
        grad[k] = (f(theta + h * e_k) - f(theta - h * e_k)) / (2.0 * h)
    return grad


def numerical_gradient(f, theta, h=1e-5, scheme="centered"):
    """
    Interface unifiée pour les 3 schémas.

    scheme : "forward" | "backward" | "centered"
    """
    schemes = {
        "forward":  numerical_gradient_forward,
        "backward": numerical_gradient_backward,
        "centered": numerical_gradient_centered,
    }
    if scheme not in schemes:
        raise ValueError(f"scheme doit être parmi {list(schemes.keys())}")
    return schemes[scheme](f, theta, h=h)


# ===========================================================================
# 2. COMPARAISON NUMÉRIQUE vs ANALYTIQUE
# ===========================================================================

def compare_gradients(theta, X, y, d, h=1e-5):
    """
    Compare les 3 gradients numériques avec le gradient analytique.

    Retourne
    --------
    report : dict avec erreurs L2 de chaque schéma par rapport à l'analytique
    """
    f = lambda th: mse(th, X, y, d)
    g_analytic  = gradient_mse(theta, X, y, d)
    g_forward   = numerical_gradient_forward(f, theta, h)
    g_backward  = numerical_gradient_backward(f, theta, h)
    g_centered  = numerical_gradient_centered(f, theta, h)

    report = {
        "analytique": g_analytic,
        "forward":    g_forward,
        "backward":   g_backward,
        "centered":   g_centered,
        "err_forward":  float(np.linalg.norm(g_forward  - g_analytic)),
        "err_backward": float(np.linalg.norm(g_backward - g_analytic)),
        "err_centered": float(np.linalg.norm(g_centered - g_analytic)),
    }
    return report


def print_gradient_comparison(report):
    print(f"{'Schéma':<12} {'Erreur L2 vs analytique':>25}")
    print("-" * 40)
    for scheme in ("forward", "backward", "centered"):
        err = report[f"err_{scheme}"]
        print(f"{scheme:<12} {err:>25.2e}")


# ===========================================================================
# 3. STABILITÉ SELON h
# ===========================================================================

def stability_vs_h(theta, X, y, d, h_values=None, scheme="centered"):
    """
    Calcule l'erreur du gradient numérique pour différentes valeurs de h.

    Retourne
    --------
    h_values : np.ndarray
    errors   : np.ndarray — erreur L2 par rapport au gradient analytique
    """
    if h_values is None:
        h_values = np.logspace(-10, 0, 50)

    f = lambda th: mse(th, X, y, d)
    g_ref = gradient_mse(theta, X, y, d)
    errors = []
    for h in h_values:
        g_num = numerical_gradient(f, theta, h=h, scheme=scheme)
        errors.append(np.linalg.norm(g_num - g_ref))
    return np.array(h_values), np.array(errors)


def plot_stability_vs_h(theta, X, y, d):
    """
    Trace les courbes d'erreur pour les 3 schémas en fonction de h.
    """
    h_values = np.logspace(-12, 0, 60)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"forward": "#378ADD", "backward": "#D85A30", "centered": "#1D9E75"}
    for scheme, color in colors.items():
        _, errors = stability_vs_h(theta, X, y, d,
                                   h_values=h_values, scheme=scheme)
        ax.loglog(h_values, errors, label=scheme, linewidth=2, color=color)
    ax.set_xlabel("h")
    ax.set_ylabel("Erreur L2 vs gradient analytique")
    ax.set_title("Stabilité du gradient numérique selon h")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


# ===========================================================================
# 4. DESCENTE DE GRADIENT AVEC GRADIENT NUMÉRIQUE
# ===========================================================================

def gradient_descent_numerical(X, y, d, theta0=None, alpha=0.01,
                                n_iter=1000, h=1e-5, scheme="centered",
                                line_search=None, project_fn=None,
                                store_every=1):
    """
    Descente de gradient utilisant le gradient numérique.

    Paramètres
    ----------
    line_search : None | "armijo" | "goldstein" | "wolfe"
                  Si None, pas constant alpha.
    project_fn  : fonction de projection (optionnel)
    scheme      : schéma numérique utilisé

    Retourne
    --------
    theta   : np.ndarray — paramètres finaux
    history : dict avec clés 'cost', 'grad_norm', 'alpha'
    """
    theta = np.zeros(d + 1) if theta0 is None else theta0.copy()
    f = lambda th: mse(th, X, y, d)
    history = {"cost": [], "grad_norm": [], "alpha": []}

    for t in range(n_iter):
        g = numerical_gradient(f, theta, h=h, scheme=scheme)
        direction = -g

        # Choix du pas
        if line_search == "armijo":
            step = armijo(f, theta, direction, g, alpha0=alpha)
        elif line_search == "goldstein":
            step = goldstein(f, theta, direction, g, alpha0=alpha)
        elif line_search == "wolfe":
            grad_fn = lambda th: numerical_gradient(f, th, h=h, scheme=scheme)
            step = wolfe(f, grad_fn, theta, direction, g, alpha0=alpha)
        else:
            step = alpha

        theta = theta + step * direction

        if project_fn is not None:
            theta = project_fn(theta)

        if t % store_every == 0:
            history["cost"].append(f(theta))
            history["grad_norm"].append(float(np.linalg.norm(g)))
            history["alpha"].append(step)

    return theta, history


# ===========================================================================
# 5. COMPARAISON DES STRATÉGIES DE PAS / LINE SEARCH
# ===========================================================================

def compare_line_searches(X, y, d, theta0=None, alpha0=0.01,
                           n_iter=500, h=1e-5, scheme="centered"):
    """
    Lance la descente avec les 4 stratégies et retourne leurs historiques.

    Retourne
    --------
    results : dict {nom_strategie: history}
    """
    strategies = {
        "Pas constant":  None,
        "Armijo":        "armijo",
        "Goldstein":     "goldstein",
        "Wolfe":         "wolfe",
    }
    results = {}
    for name, ls in strategies.items():
        theta0_ = np.zeros(d + 1) if theta0 is None else theta0.copy()
        _, history = gradient_descent_numerical(
            X, y, d, theta0=theta0_, alpha=alpha0,
            n_iter=n_iter, h=h, scheme=scheme, line_search=ls
        )
        results[name] = history
    return results


def plot_line_search_comparison(results):
    """
    Trace les courbes de coût et les pas pour toutes les stratégies.
    """
    colors = ["#378ADD", "#D85A30", "#1D9E75", "#D4537E"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for i, (name, history) in enumerate(results.items()):
        c = colors[i % len(colors)]
        axes[0].plot(history["cost"], label=name, color=c, linewidth=2)
        axes[1].plot(history["alpha"], label=name, color=c,
                     linewidth=1.5, alpha=0.8)

    axes[0].set_title("Évolution du coût MSE")
    axes[0].set_xlabel("Itération")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].set_title("Évolution du pas αt")
    axes[1].set_xlabel("Itération")
    axes[1].set_ylabel("αt")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig


# ===========================================================================
# 6. ÉTUDE DES PAS : FIXE / DÉCROISSANT / ADAPTATIF
# ===========================================================================

def decaying_step(alpha0, t, beta=0.5):
    """
    Pas décroissant : alpha_t = alpha0 / (t+1)^beta

    Conditions classiques (Robbins-Monro) :
      sum alpha_t = inf  et  sum alpha_t² < inf  ⟺  0.5 < beta ≤ 1
    """
    return alpha0 / ((t + 1) ** beta)


def gradient_descent_decaying(X, y, d, theta0=None, alpha0=0.1,
                               beta=0.5, n_iter=1000, h=1e-5,
                               scheme="centered"):
    """
    Descente avec pas décroissant alpha_t = alpha0 / (t+1)^beta.
    """
    theta = np.zeros(d + 1) if theta0 is None else theta0.copy()
    f = lambda th: mse(th, X, y, d)
    history = {"cost": [], "grad_norm": [], "alpha": []}

    for t in range(n_iter):
        g = numerical_gradient(f, theta, h=h, scheme=scheme)
        step = decaying_step(alpha0, t, beta=beta)
        theta = theta - step * g
        history["cost"].append(f(theta))
        history["grad_norm"].append(float(np.linalg.norm(g)))
        history["alpha"].append(step)

    return theta, history


def compare_step_strategies(X, y, d, theta0=None, alpha0=0.05,
                              n_iter=500, h=1e-5):
    """
    Compare pas fixe, décroissant (beta=0.5), décroissant (beta=1.0)
    et Armijo sur un même problème.
    """
    results = {}

    # Pas constant
    th0 = np.zeros(d + 1) if theta0 is None else theta0.copy()
    _, hist = gradient_descent_numerical(X, y, d, theta0=th0,
                                          alpha=alpha0, n_iter=n_iter, h=h)
    results["Fixe"] = hist

    # Décroissant beta=0.5
    th0 = np.zeros(d + 1) if theta0 is None else theta0.copy()
    _, hist = gradient_descent_decaying(X, y, d, theta0=th0,
                                         alpha0=alpha0, beta=0.5,
                                         n_iter=n_iter, h=h)
    results["Décroissant β=0.5"] = hist

    # Décroissant beta=1.0
    th0 = np.zeros(d + 1) if theta0 is None else theta0.copy()
    _, hist = gradient_descent_decaying(X, y, d, theta0=th0,
                                         alpha0=alpha0, beta=1.0,
                                         n_iter=n_iter, h=h)
    results["Décroissant β=1.0"] = hist

    # Armijo
    th0 = np.zeros(d + 1) if theta0 is None else theta0.copy()
    _, hist = gradient_descent_numerical(X, y, d, theta0=th0,
                                          alpha=alpha0, n_iter=n_iter,
                                          h=h, line_search="armijo")
    results["Armijo"] = hist

    return results