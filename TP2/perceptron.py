"""
perceptron.py — Classification binaire par Perceptron (TP2)
============================================================
Contenu :
  1. Modèle linéaire f(x) = w^T x + b
  2. Transformation polynomiale phi_d pour données non séparables
  3. Fonction de coût du perceptron + sous-gradient
  4. Descente par sous-gradient (batch)
  5. Descente avec line search (Armijo / Goldstein / Wolfe)
  6. Stratégies de pas : fixe, décroissant, adaptatif, self-adaptatif
  7. Régularisation Ridge (L2)
  8. Visualisation
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath('..'))
from utils import (
    armijo, goldstein, wolfe,
    l2_regularization, project_l2_ball,
    SelfAdaptiveLineSearch
)


# ===========================================================================
# 1. TRANSFORMATION POLYNOMIALE (pour données non linéairement séparables)
# ===========================================================================

def poly_features(X, d):
    """
    Transformation polynomiale des features jusqu'au degré d.
    Utilisée pour rendre des données non linéairement séparables.

    Paramètres
    ----------
    X : np.ndarray shape (n, p)
    d : int — degré maximal

    Retourne
    --------
    Phi : np.ndarray shape (n, nb_features)
    """
    if d == 1:
        return X.copy()
    from itertools import combinations_with_replacement
    n, p = X.shape
    features = [X]
    for deg in range(2, d + 1):
        for combo in combinations_with_replacement(range(p), deg):
            feat = np.prod(X[:, combo], axis=1, keepdims=True)
            features.append(feat)
    return np.hstack(features)


# ===========================================================================
# 2. MODÈLE LINÉAIRE
# ===========================================================================

def predict_score(w, b, X):
    """
    Score linéaire f(x) = w^T x + b pour chaque exemple.

    Retourne
    --------
    scores : np.ndarray shape (n,)
    """
    return X @ w + b


def predict(w, b, X):
    """
    Prédiction binaire ŷ = sign(f(x)).
    Les scores nuls sont classés +1 par convention.

    Retourne
    --------
    y_pred : np.ndarray shape (n,) avec valeurs dans {-1, +1}
    """
    scores = predict_score(w, b, X)
    return np.where(scores >= 0, 1, -1).astype(int)


# ===========================================================================
# 3. FONCTION DE COÛT DU PERCEPTRON
# ===========================================================================

def perceptron_loss_individual(w, b, xi, yi):
    """
    Perte individuelle : ℓ(w,b; xi,yi) = max(0, -yi*(w^T xi + b))
    """
    return float(max(0.0, -yi * (np.dot(w, xi) + b)))


def perceptron_loss(w, b, X, y):
    """
    Coût global : J(w,b) = (1/n) * sum_i max(0, -yi*(w^T xi + b))
    """
    margins = y * (X @ w + b)
    return float(np.mean(np.maximum(0.0, -margins)))


def hinge_loss_individual(w, b, xi, yi):
    """
    Perte hinge individuelle : ℓ(w,b; xi,yi) = max(0, 1 - yi*(w^T xi + b))
    (utilisée dans PA et OSD — TP5)
    """
    return float(max(0.0, 1.0 - yi * (np.dot(w, xi) + b)))


def hinge_loss(w, b, X, y):
    """
    Coût hinge global : J(w,b) = (1/n) * sum_i max(0, 1 - yi*(w^T xi + b))
    """
    margins = y * (X @ w + b)
    return float(np.mean(np.maximum(0.0, 1.0 - margins)))


# ===========================================================================
# 4. SOUS-GRADIENT
# ===========================================================================

def subgradient_individual(w, b, xi, yi):
    """
    Sous-gradient de ℓ_i = max(0, -yi*(w^T xi + b)) par rapport à (w, b).

    ∂w ℓ_i = -yi*xi  si yi*(w^T xi + b) < 0
              0        sinon
    ∂b ℓ_i = -yi      si yi*(w^T xi + b) < 0
              0        sinon

    Retourne
    --------
    gw : np.ndarray shape (p,)
    gb : float
    """
    margin = yi * (np.dot(w, xi) + b)
    if margin < 0:
        return -yi * xi, float(-yi)
    else:
        return np.zeros_like(w), 0.0


def subgradient_batch(w, b, X, y):
    """
    Sous-gradient global de J(w,b) = (1/n) * sum_i ℓ_i

    Retourne
    --------
    gw : np.ndarray shape (p,)
    gb : float
    """
    n = len(y)
    margins = y * (X @ w + b)
    mask = margins < 0                       # exemples mal classés
    gw = -np.mean(y[mask, None] * X[mask], axis=0) if mask.any() else np.zeros(w.shape)
    gb = float(-np.mean(y[mask])) if mask.any() else 0.0
    return gw, gb


def subgradient_hinge_individual(w, b, xi, yi):
    """
    Sous-gradient de la perte hinge individuelle max(0, 1 - yi*(w^T xi+b)).
    Utilisé dans OSD / TP5.
    """
    margin = yi * (np.dot(w, xi) + b)
    if margin < 1.0:
        return -yi * xi, float(-yi)
    else:
        return np.zeros_like(w), 0.0


def subgradient_hinge_batch(w, b, X, y):
    """
    Sous-gradient global de la perte hinge.
    """
    margins = y * (X @ w + b)
    mask = margins < 1.0
    gw = -np.mean(y[mask, None] * X[mask], axis=0) if mask.any() else np.zeros(w.shape)
    gb = float(-np.mean(y[mask])) if mask.any() else 0.0
    return gw, gb


# ===========================================================================
# 5. DESCENTE PAR SOUS-GRADIENT (batch) — méthode de référence
# ===========================================================================

def subgradient_descent(X, y, alpha=0.01, n_iter=1000,
                         w0=None, b0=0.0, project_fn=None,
                         loss="perceptron", store_every=1):
    """
    Descente par sous-gradient standard (batch).

    Paramètres
    ----------
    alpha      : float — pas constant
    project_fn : callable(w) → w_projected (optionnel)
    loss       : "perceptron" | "hinge"
    store_every: fréquence d'enregistrement

    Retourne
    --------
    w, b    : paramètres finaux
    history : dict avec 'cost', 'grad_norm', 'accuracy'
    """
    p = X.shape[1]
    w = np.zeros(p) if w0 is None else w0.copy()
    b = float(b0)
    history = {"cost": [], "grad_norm": [], "accuracy": []}

    loss_fn  = perceptron_loss  if loss == "perceptron" else hinge_loss
    grad_fn  = subgradient_batch if loss == "perceptron" else subgradient_hinge_batch

    for t in range(n_iter):
        gw, gb = grad_fn(w, b, X, y)
        w = w - alpha * gw
        b = b - alpha * gb
        if project_fn is not None:
            w = project_fn(w)

        if t % store_every == 0:
            history["cost"].append(loss_fn(w, b, X, y))
            history["grad_norm"].append(float(np.linalg.norm(np.append(gw, gb))))
            history["accuracy"].append(float(np.mean(predict(w, b, X) == y)))

    return w, b, history


# ===========================================================================
# 6. DESCENTE AVEC LINE SEARCH
# ===========================================================================

def subgradient_descent_linesearch(X, y, alpha0=1.0, n_iter=500,
                                    w0=None, b0=0.0,
                                    line_search="armijo", loss="perceptron"):
    """
    Descente par sous-gradient avec line search.
    Paramètre theta = (w, b) vectorisé pour s'interfacer avec utils.py.
    """
    p = X.shape[1]
    theta = np.zeros(p + 1) if w0 is None else np.append(w0, b0)

    loss_fn = perceptron_loss if loss == "perceptron" else hinge_loss
    grad_fn = subgradient_batch if loss == "perceptron" else subgradient_hinge_batch

    def f(th):
        return loss_fn(th[:-1], th[-1], X, y)

    def grad_f(th):
        gw, gb = grad_fn(th[:-1], th[-1], X, y)
        return np.append(gw, gb)

    history = {"cost": [], "grad_norm": [], "alpha": []}

    for _ in range(n_iter):
        g = grad_f(theta)
        direction = -g

        if line_search == "armijo":
            step = armijo(f, theta, direction, g, alpha0=alpha0)
        elif line_search == "goldstein":
            step = goldstein(f, theta, direction, g, alpha0=alpha0)
        elif line_search == "wolfe":
            step = wolfe(f, grad_f, theta, direction, g, alpha0=alpha0)
        else:
            step = alpha0

        theta = theta + step * direction
        history["cost"].append(f(theta))
        history["grad_norm"].append(float(np.linalg.norm(g)))
        history["alpha"].append(step)

    w, b = theta[:-1], float(theta[-1])
    return w, b, history


# ===========================================================================
# 7. STRATÉGIES DE PAS
# ===========================================================================

def decaying_step(alpha0, t, beta=0.5):
    """alpha_t = alpha0 / (t+1)^beta"""
    return alpha0 / ((t + 1) ** beta)


def subgradient_descent_decaying(X, y, alpha0=0.1, beta=0.5,
                                  n_iter=1000, w0=None, b0=0.0,
                                  loss="perceptron"):
    """Descente avec pas décroissant."""
    p = X.shape[1]
    w = np.zeros(p) if w0 is None else w0.copy()
    b = float(b0)
    loss_fn = perceptron_loss if loss == "perceptron" else hinge_loss
    grad_fn = subgradient_batch if loss == "perceptron" else subgradient_hinge_batch
    history = {"cost": [], "grad_norm": [], "alpha": [], "accuracy": []}

    for t in range(n_iter):
        gw, gb = grad_fn(w, b, X, y)
        step = decaying_step(alpha0, t, beta)
        w = w - step * gw
        b = b - step * gb
        history["cost"].append(loss_fn(w, b, X, y))
        history["grad_norm"].append(float(np.linalg.norm(np.append(gw, gb))))
        history["alpha"].append(step)
        history["accuracy"].append(float(np.mean(predict(w, b, X) == y)))

    return w, b, history


def subgradient_descent_adaptive(X, y, alpha0=0.1, n_iter=1000,
                                  w0=None, b0=0.0,
                                  success_threshold=0.001,
                                  increase=1.05, decrease=0.7,
                                  loss="perceptron"):
    """Descente avec pas adaptatif (augmente si succès, diminue sinon)."""
    p = X.shape[1]
    w = np.zeros(p) if w0 is None else w0.copy()
    b = float(b0)
    alpha = alpha0
    loss_fn = perceptron_loss if loss == "perceptron" else hinge_loss
    grad_fn = subgradient_batch if loss == "perceptron" else subgradient_hinge_batch
    history = {"cost": [], "grad_norm": [], "alpha": [], "accuracy": []}

    for _ in range(n_iter):
        gw, gb = grad_fn(w, b, X, y)
        cost_before = loss_fn(w, b, X, y)
        w_new = w - alpha * gw
        b_new = b - alpha * gb
        cost_after = loss_fn(w_new, b_new, X, y)
        rel_drop = (cost_before - cost_after) / (abs(cost_before) + 1e-12)
        if rel_drop >= success_threshold:
            alpha *= increase
        else:
            alpha *= decrease
        w, b = w_new, b_new
        history["cost"].append(cost_after)
        history["grad_norm"].append(float(np.linalg.norm(np.append(gw, gb))))
        history["alpha"].append(alpha)
        history["accuracy"].append(float(np.mean(predict(w, b, X) == y)))

    return w, b, history


def compare_step_strategies(X, y, alpha0=0.1, n_iter=500, loss="perceptron"):
    """
    Lance les 4 stratégies de pas sur un même problème.

    Retourne
    --------
    results : dict {nom: history}
    """
    p = X.shape[1]
    w0 = np.zeros(p)
    results = {}

    # Pas constant
    _, _, hist = subgradient_descent(X, y, alpha=alpha0, n_iter=n_iter,
                                      w0=w0.copy(), loss=loss)
    results["Fixe"] = hist

    # Décroissant β=0.5
    _, _, hist = subgradient_descent_decaying(X, y, alpha0=alpha0, beta=0.5,
                                               n_iter=n_iter, w0=w0.copy(), loss=loss)
    results["Décroissant β=0.5"] = hist

    # Décroissant β=1.0
    _, _, hist = subgradient_descent_decaying(X, y, alpha0=alpha0, beta=1.0,
                                               n_iter=n_iter, w0=w0.copy(), loss=loss)
    results["Décroissant β=1.0"] = hist

    # Adaptatif
    _, _, hist = subgradient_descent_adaptive(X, y, alpha0=alpha0,
                                               n_iter=n_iter, w0=w0.copy(), loss=loss)
    results["Adaptatif"] = hist

    return results


def compare_line_searches(X, y, alpha0=0.5, n_iter=300, loss="perceptron"):
    """Compare Armijo, Goldstein, Wolfe et pas constant."""
    p = X.shape[1]
    w0 = np.zeros(p)
    results = {}
    for ls in ["fixed", "armijo", "goldstein", "wolfe"]:
        name = ls.capitalize() if ls != "fixed" else "Pas constant"
        _, _, hist = subgradient_descent_linesearch(
            X, y, alpha0=alpha0, n_iter=n_iter,
            w0=w0.copy(), line_search=ls, loss=loss
        )
        results[name] = hist
    return results


# ===========================================================================
# 8. RÉGULARISATION RIDGE (L2)
# ===========================================================================

def subgradient_descent_ridge(X, y, lambda_=0.01, alpha=0.01,
                               n_iter=1000, w0=None, b0=0.0,
                               loss="perceptron"):
    """
    Descente par sous-gradient avec régularisation Ridge.
    Jλ(w,b) = J(w,b) + (λ/2)||w||²
    ∇Jλ = ∇J + λw  (biais non régularisé)
    """
    p = X.shape[1]
    w = np.zeros(p) if w0 is None else w0.copy()
    b = float(b0)
    loss_fn = perceptron_loss if loss == "perceptron" else hinge_loss
    grad_fn = subgradient_batch if loss == "perceptron" else subgradient_hinge_batch
    history = {"cost": [], "reg_cost": [], "accuracy": []}

    for _ in range(n_iter):
        gw, gb = grad_fn(w, b, X, y)
        _, g_reg = l2_regularization(w, lambda_)
        w = w - alpha * (gw + g_reg)
        b = b - alpha * gb
        history["cost"].append(loss_fn(w, b, X, y))
        history["reg_cost"].append(
            loss_fn(w, b, X, y) + 0.5 * lambda_ * np.dot(w, w)
        )
        history["accuracy"].append(float(np.mean(predict(w, b, X) == y)))

    return w, b, history


# ===========================================================================
# 9. VISUALISATION
# ===========================================================================

def plot_decision_boundary(w, b, X, y, d=1, title="Frontière de décision"):
    """
    Trace la frontière de décision pour p=2 (ou les 2 premières dimensions).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {-1: "#378ADD", 1: "#D85A30"}
    markers = {-1: "o", 1: "s"}
    for label in [-1, 1]:
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=colors[label], marker=markers[label],
                   alpha=0.6, s=25, label=f"Classe {label}",
                   edgecolors="none")

    # Grille de décision
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if d > 1:
        grid = poly_features(grid, d)
    zz = predict(w, b, grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.12,
                colors=["#378ADD", "#D85A30"])
    ax.contour(xx, yy, zz, colors="black", linewidths=1.2)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_convergence(history, title="Convergence — perceptron"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(history["cost"], color="#378ADD", linewidth=2)
    axes[0].set_title("Coût")
    axes[0].set_xlabel("Itération"); axes[0].set_ylabel("J(w,b)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(history["grad_norm"], color="#D85A30", linewidth=2)
    axes[1].set_title("Norme du sous-gradient")
    axes[1].set_xlabel("Itération"); axes[1].set_ylabel("||g||")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    if "accuracy" in history:
        axes[2].plot(history["accuracy"], color="#1D9E75", linewidth=2)
        axes[2].set_title("Accuracy")
        axes[2].set_xlabel("Itération"); axes[2].set_ylabel("Acc.")
        axes[2].set_ylim(0, 1.05)
        axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(title, y=1.01)
    plt.tight_layout()
    return fig


def plot_multi_histories(results_dict, metric="cost",
                          title="Comparaison"):
    colors = ["#378ADD", "#D85A30", "#1D9E75", "#D4537E",
              "#EF9F27", "#7F77DD"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, hist) in enumerate(results_dict.items()):
        ax.plot(hist[metric], label=name, linewidth=2,
                color=colors[i % len(colors)])
    ax.set_xlabel("Itération")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig