"""
polynomial.py — Modèle de régression polynomiale (TP1)
=======================================================
Contenu :
  1. Transformation phi_d(x) : x → (1, x, x², ..., x^d)
  2. Prédiction ŷ = theta @ phi_d(x)
  3. Coût MSE  Jd(theta)
  4. Gradient analytique de Jd
  5. Utilitaires : plot du modèle, courbes biais-variance
"""

import numpy as np
import matplotlib.pyplot as plt


# ===========================================================================
# 1. TRANSFORMATION POLYNOMIALE
# ===========================================================================

def phi(x, d):
    """
    Transforme un scalaire ou vecteur x en features polynomiales.

    phi_d(x) = [1, x, x², ..., x^d]  — shape (d+1,) si x est un scalaire
                                        shape (n, d+1) si x est un vecteur

    Paramètres
    ----------
    x : float ou np.ndarray shape (n,)
    d : int — degré du polynôme

    Retourne
    --------
    Phi : np.ndarray shape (d+1,) ou (n, d+1)
    """
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    Phi = np.column_stack([x ** k for k in range(d + 1)])  # shape (n, d+1)
    return Phi[0] if scalar else Phi


# ===========================================================================
# 2. PRÉDICTION
# ===========================================================================

def predict(theta, x, d):
    """
    ŷ = Phi @ theta

    Paramètres
    ----------
    theta : np.ndarray shape (d+1,)
    x     : float ou np.ndarray shape (n,)
    d     : int

    Retourne
    --------
    y_hat : float ou np.ndarray shape (n,)
    """
    Phi = phi(x, d)
    return Phi @ theta


# ===========================================================================
# 3. COÛT MSE
# ===========================================================================

def mse(theta, X, y, d):
    """
    Jd(theta) = (1/n) * sum_i (ŷ(x_i) - y_i)²

    Paramètres
    ----------
    theta : np.ndarray shape (d+1,)
    X     : np.ndarray shape (n,)  — features 1D
    y     : np.ndarray shape (n,)  — cibles
    d     : int

    Retourne
    --------
    cost : float
    """
    y_hat = predict(theta, X, d)
    residuals = y_hat - y
    return float(np.mean(residuals ** 2))


def mse_individual(theta, xi, yi, d):
    """
    Perte instantanée sur un seul exemple (pour le cadre online / SGD).
    ℓ(theta; xi, yi) = (ŷ(xi) - yi)²
    """
    y_hat = predict(theta, xi, d)
    return float((y_hat - yi) ** 2)


# ===========================================================================
# 4. GRADIENT ANALYTIQUE
# ===========================================================================

def gradient_mse(theta, X, y, d):
    """
    Gradient analytique de Jd(theta) par rapport à theta.

    ∇Jd(theta) = (2/n) * Phi^T (Phi @ theta - y)

    Retourne
    --------
    grad : np.ndarray shape (d+1,)
    """
    Phi = phi(X, d)           # (n, d+1)
    residuals = Phi @ theta - y  # (n,)
    return (2.0 / len(y)) * Phi.T @ residuals


def gradient_mse_single(theta, xi, yi, d):
    """
    Gradient de la perte instantanée ℓ(theta; xi, yi) — pour online / SGD.

    ∇ℓ = 2 * (ŷ(xi) - yi) * phi_d(xi)
    """
    phi_i = phi(xi, d)             # (d+1,)
    residual = phi_i @ theta - yi
    return 2.0 * residual * phi_i


def gradient_mse_batch(theta, X_batch, y_batch, d):
    """
    Gradient sur un mini-lot (pour SGD avec mini-batch).
    """
    Phi = phi(X_batch, d)
    residuals = Phi @ theta - y_batch
    return (2.0 / len(y_batch)) * Phi.T @ residuals


# ===========================================================================
# 5. DESCENTE DE GRADIENT (batch) — méthode de référence TP1 + TP3
# ===========================================================================

def gradient_descent(X, y, d, theta0=None, alpha=0.01, n_iter=1000,
                     project_fn=None, store_every=1):
    """
    Descente de gradient standard (batch).

    Paramètres
    ----------
    X, y        : données
    d           : degré
    theta0      : initialisation (None → zéros)
    alpha       : pas constant
    n_iter      : nombre d'itérations
    project_fn  : fonction de projection optionnelle theta → theta_projected
    store_every : fréquence d'enregistrement des métriques

    Retourne
    --------
    theta  : np.ndarray — paramètres finaux
    history : dict avec clés 'cost', 'grad_norm', 'theta_list'
    """
    theta = np.zeros(d + 1) if theta0 is None else theta0.copy()
    history = {"cost": [], "grad_norm": [], "theta_list": []}

    for t in range(n_iter):
        g = gradient_mse(theta, X, y, d)
        theta = theta - alpha * g
        if project_fn is not None:
            theta = project_fn(theta)
        if t % store_every == 0:
            history["cost"].append(mse(theta, X, y, d))
            history["grad_norm"].append(float(np.linalg.norm(g)))
            history["theta_list"].append(theta.copy())

    return theta, history


def gradient_descent_linesearch(X, y, d, theta0=None, line_search_fn=None,
                                 n_iter=500, project_fn=None):
    """
    Descente de gradient avec line search (Armijo / Goldstein / Wolfe).

    line_search_fn : callable(f, theta, d_dir, g) → alpha
                     doit être l'une des fonctions de utils.py
    """
    from functools import partial

    theta = np.zeros(d + 1) if theta0 is None else theta0.copy()
    history = {"cost": [], "grad_norm": [], "alpha": []}

    f = lambda th: mse(th, X, y, d)

    for _ in range(n_iter):
        g = gradient_mse(theta, X, y, d)
        direction = -g
        if line_search_fn is not None:
            alpha = line_search_fn(f, theta, direction, g)
        else:
            alpha = 0.01
        theta = theta + alpha * direction
        if project_fn is not None:
            theta = project_fn(theta)
        history["cost"].append(f(theta))
        history["grad_norm"].append(float(np.linalg.norm(g)))
        history["alpha"].append(alpha)

    return theta, history


# ===========================================================================
# 6. MÉTRIQUES TRAIN / TEST
# ===========================================================================

def mse_score(theta, X, y, d):
    return mse(theta, X, y, d)


def r2_score(theta, X, y, d):
    """
    R² = 1 - SS_res / SS_tot
    """
    y_hat = predict(theta, X, d)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def bias_variance_curve(X_train, y_train, X_test, y_test,
                        degrees, alpha=0.01, n_iter=2000):
    """
    Calcule MSE train et test pour plusieurs degrés d.

    Retourne
    --------
    train_errors, test_errors : lists de float
    """
    train_errors, test_errors = [], []
    for d in degrees:
        theta0 = np.zeros(d + 1)
        theta, _ = gradient_descent(X_train, y_train, d,
                                    theta0=theta0, alpha=alpha,
                                    n_iter=n_iter)
        train_errors.append(mse(theta, X_train, y_train, d))
        test_errors.append(mse(theta, X_test, y_test, d))
    return train_errors, test_errors


# ===========================================================================
# 7. VISUALISATION
# ===========================================================================

def plot_polynomial_fit(X, y, theta, d, title=None):
    """
    Trace les données et la courbe ajustée.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(X, y, alpha=0.5, s=20, color="#378ADD", label="Données")
    x_line = np.linspace(X.min(), X.max(), 300)
    y_line = predict(theta, x_line, d)
    ax.plot(x_line, y_line, color="#D85A30", linewidth=2,
            label=f"Polynôme degré {d}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or f"Régression polynomiale — degré {d}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_bias_variance(degrees, train_errors, test_errors):
    """
    Courbe biais-variance : MSE train et test en fonction du degré.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(degrees, train_errors, marker='o', linewidth=2,
            color="#378ADD", label="MSE Train")
    ax.plot(degrees, test_errors, marker='s', linewidth=2,
            color="#D85A30", label="MSE Test")
    ax.set_xlabel("Degré d")
    ax.set_ylabel("MSE")
    ax.set_title("Courbe biais-variance")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    best_d = degrees[int(np.argmin(test_errors))]
    ax.axvline(best_d, linestyle=":", color="#1D9E75",
               label=f"d* = {best_d}")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_convergence(history, title="Convergence — coût MSE"):
    """
    Trace l'évolution du coût au fil des itérations.
    history : dict avec clé 'cost'
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["cost"], color="#378ADD", linewidth=2)
    axes[0].set_xlabel("Itération")
    axes[0].set_ylabel("MSE")
    axes[0].set_title(title)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(history["grad_norm"], color="#D85A30", linewidth=2)
    axes[1].set_xlabel("Itération")
    axes[1].set_ylabel("||∇J||")
    axes[1].set_title("Norme du gradient")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig