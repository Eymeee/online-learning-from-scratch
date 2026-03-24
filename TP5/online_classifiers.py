"""
online_classifiers.py — Classificateurs en ligne supervisés (TP5)
==================================================================
Contenu :
  1. Perceptron standard en ligne
  2. Normalized Perceptron
  3. Passive-Aggressive : PA, PA-I, PA-II
  4. Online Subgradient Descent (OSD) avec régularisation L1/L2
  5. Comparaison unifiée
  6. Visualisation
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath('..'))
from utils import project_l2_ball, apply_l1_update, apply_l2_update
from TP2.perceptron import subgradient_hinge_individual


# ===========================================================================
# UTILITAIRES COMMUNS
# ===========================================================================

def _init_history():
    return {
        "cumul_errors": [],
        "instant_loss": [],
        "cumul_loss":   [],
        "w_norm":       [],
    }


def _update_history(hist, err, loss, cumul_err, cumul_loss, w):
    hist["cumul_errors"].append(cumul_err)
    hist["instant_loss"].append(loss)
    hist["cumul_loss"].append(cumul_loss)
    hist["w_norm"].append(float(np.linalg.norm(w)))


# ===========================================================================
# 1. PERCEPTRON STANDARD EN LIGNE
# ===========================================================================

def perceptron_online(X, y, w0=None, b0=0.0):
    """
    Perceptron standard en ligne.

    Si ŷt != yt : wt+1 = wt + yt*xt
    Sinon       : wt+1 = wt

    Borne : MT <= (R/gamma)^2  dans le cas séparable avec marge gamma.
    """
    n, p = X.shape
    w = np.zeros(p) if w0 is None else w0.copy()
    b = float(b0)
    hist = _init_history()
    cumul_err, cumul_loss = 0, 0.0

    for t in range(n):
        score = np.dot(w, X[t]) + b
        y_hat = 1 if score >= 0 else -1
        err = int(y_hat != y[t])
        loss = max(0.0, -y[t] * score)
        cumul_err += err
        cumul_loss += loss

        if err:
            w = w + y[t] * X[t]
            b = b + y[t]

        _update_history(hist, err, loss, cumul_err, cumul_loss, w)

    return w, b, hist


# ===========================================================================
# 2. NORMALIZED PERCEPTRON
# ===========================================================================

def normalized_perceptron_online(X, y, w0=None, b0=0.0):
    """
    Normalized Perceptron.

    Mise à jour en cas d'erreur :
      wt+1 = wt + yt * xt / ||xt||

    Réduit l'effet des exemples de grande norme.
    """
    n, p = X.shape
    w = np.zeros(p) if w0 is None else w0.copy()
    b = float(b0)
    hist = _init_history()
    cumul_err, cumul_loss = 0, 0.0

    for t in range(n):
        score = np.dot(w, X[t]) + b
        y_hat = 1 if score >= 0 else -1
        err = int(y_hat != y[t])
        loss = max(0.0, -y[t] * score)
        cumul_err += err
        cumul_loss += loss

        if err:
            norm_xt = np.linalg.norm(X[t])
            x_norm = X[t] / (norm_xt + 1e-12)
            w = w + y[t] * x_norm
            b = b + y[t]

        _update_history(hist, err, loss, cumul_err, cumul_loss, w)

    return w, b, hist


# ===========================================================================
# 3. PASSIVE-AGGRESSIVE
# ===========================================================================

def _pa_tau(loss_t, norm_xt_sq, C=1.0, variant="PA"):
    """
    τt selon la variante PA.
      PA   : τt = ℓt / ||xt||²
      PA-I : τt = min(C, ℓt / ||xt||²)
      PA-II: τt = ℓt / (||xt||² + 1/(2C))
    """
    base = loss_t / (norm_xt_sq + 1e-12)
    if variant == "PA":
        return base
    elif variant == "PA-I":
        return min(C, base)
    elif variant == "PA-II":
        return loss_t / (norm_xt_sq + 1.0 / (2.0 * C + 1e-12))
    raise ValueError("variant doit être PA, PA-I ou PA-II")


def passive_aggressive_online(X, y, C=1.0, variant="PA", w0=None, b0=0.0):
    """
    Passive-Aggressive Online Learning.

    Perte hinge : ℓt = max(0, 1 - yt*(wt^T xt + b))
    Mise à jour : wt+1 = wt + τt * yt * xt

    Passif  : si ℓt = 0 → aucune mise à jour
    Agressif: sinon → mise à jour minimale pour annuler la perte
    """
    n, p = X.shape
    w = np.zeros(p) if w0 is None else w0.copy()
    b = float(b0)
    hist = _init_history()
    hist["tau"] = []
    cumul_err, cumul_loss = 0, 0.0

    for t in range(n):
        score = np.dot(w, X[t]) + b
        y_hat = 1 if score >= 0 else -1
        err = int(y_hat != y[t])
        loss_t = max(0.0, 1.0 - y[t] * score)
        cumul_err += err
        cumul_loss += loss_t

        if loss_t > 0:
            norm_sq = float(np.dot(X[t], X[t]))
            tau = _pa_tau(loss_t, norm_sq, C=C, variant=variant)
            w = w + tau * y[t] * X[t]
            b = b + tau * y[t]
            hist["tau"].append(tau)
        else:
            hist["tau"].append(0.0)

        _update_history(hist, err, loss_t, cumul_err, cumul_loss, w)

    return w, b, hist


def compare_pa_variants(X, y, C=1.0):
    """Compare PA, PA-I, PA-II sur le même flux."""
    results = {}
    for variant in ["PA", "PA-I", "PA-II"]:
        w, b, hist = passive_aggressive_online(X, y, C=C, variant=variant)
        results[variant] = (w, b, hist)
    return results


def study_C_effect(X, y, C_values=None, variant="PA-I"):
    """Étudie l'effet de C pour PA-I ou PA-II."""
    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    results = {}
    for C in C_values:
        w, b, hist = passive_aggressive_online(X, y, C=C, variant=variant)
        results[f"C={C}"] = (w, b, hist)
    return results


# ===========================================================================
# 4. ONLINE SUBGRADIENT DESCENT (OSD)
# ===========================================================================

def osd_online(X, y, eta=None, eta0=0.1, decay="sqrt",
               lambda_l2=0.0, lambda_l1=0.0,
               project_radius=None, w0=None, b0=0.0):
    """
    Online Subgradient Descent avec perte hinge.

    wt+1 = ΠS(wt - ηt*gt [- ηt*λ_reg])

    Régularisation L2 : w ← w*(1 - η*λ2)
    Régularisation L1 : w ← sign(w)*max(|w| - η*λ1, 0)  (proximal)
    """
    n, p = X.shape
    w = np.zeros(p) if w0 is None else w0.copy()
    b = float(b0)
    hist = _init_history()
    hist["sparsity"] = []
    cumul_err, cumul_loss = 0, 0.0

    for t in range(n):
        if eta is not None:
            step = eta
        elif decay == "sqrt":
            step = eta0 / np.sqrt(t + 1)
        elif decay == "poly":
            step = eta0 / (t + 1)
        else:
            step = eta0

        score = np.dot(w, X[t]) + b
        y_hat = 1 if score >= 0 else -1
        err = int(y_hat != y[t])
        loss_t = max(0.0, 1.0 - y[t] * score)
        cumul_err += err
        cumul_loss += loss_t

        gw, gb = subgradient_hinge_individual(w, b, X[t], y[t])

        w = w - step * gw
        b = b - step * gb

        if lambda_l2 > 0:
            w = apply_l2_update(w, step, lambda_l2)
        if lambda_l1 > 0:
            w = apply_l1_update(w, step, lambda_l1)
        if project_radius is not None:
            w = project_l2_ball(w, project_radius)

        _update_history(hist, err, loss_t, cumul_err, cumul_loss, w)
        hist["sparsity"].append(float(np.mean(np.abs(w) < 1e-8)))

    return w, b, hist


def study_osd_steps(X, y, eta0_values=None):
    """Étudie l'effet du pas η₀ pour OSD."""
    if eta0_values is None:
        eta0_values = [0.001, 0.01, 0.1, 0.5]
    results = {}
    for eta0 in eta0_values:
        w, b, hist = osd_online(X, y, eta0=eta0, decay="sqrt")
        results[f"η₀={eta0}"] = (w, b, hist)
    return results


def study_osd_regularization(X, y, lambdas=None, reg="l2", eta0=0.1):
    """Étudie l'effet de la régularisation L1 ou L2 pour OSD."""
    if lambdas is None:
        lambdas = [0.0, 1e-4, 1e-3, 1e-2, 0.1]
    results = {}
    for lam in lambdas:
        kw = {"lambda_l2": lam} if reg == "l2" else {"lambda_l1": lam}
        w, b, hist = osd_online(X, y, eta0=eta0, decay="sqrt", **kw)
        results[f"λ={lam}"] = (w, b, hist)
    return results


# ===========================================================================
# 5. COMPARAISON UNIFIÉE
# ===========================================================================

def compare_all_classifiers(X, y, eta0=0.1, C=1.0):
    """Lance tous les classificateurs sur le même flux."""
    p = X.shape[1]
    w0 = np.zeros(p)
    return {
        "Perceptron":       perceptron_online(X, y, w0=w0.copy()),
        "Norm. Perceptron": normalized_perceptron_online(X, y, w0=w0.copy()),
        "PA":               passive_aggressive_online(X, y, C=C,
                                variant="PA",   w0=w0.copy()),
        "PA-I":             passive_aggressive_online(X, y, C=C,
                                variant="PA-I", w0=w0.copy()),
        "PA-II":            passive_aggressive_online(X, y, C=C,
                                variant="PA-II",w0=w0.copy()),
        "OSD":              osd_online(X, y, eta0=eta0,
                                decay="sqrt", w0=w0.copy()),
        "OSD+L2":           osd_online(X, y, eta0=eta0, decay="sqrt",
                                lambda_l2=1e-3, w0=w0.copy()),
        "OSD+L1":           osd_online(X, y, eta0=eta0, decay="sqrt",
                                lambda_l1=1e-3, w0=w0.copy()),
    }


# ===========================================================================
# 6. VISUALISATION
# ===========================================================================

COLORS = {
    "Perceptron":       "#888780",
    "Norm. Perceptron": "#378ADD",
    "PA":               "#D85A30",
    "PA-I":             "#1D9E75",
    "PA-II":            "#D4537E",
    "OSD":              "#EF9F27",
    "OSD+L2":           "#7F77DD",
    "OSD+L1":           "#5DCAA5",
}


def _get_hist(vals):
    """Extrait le dict history quelle que soit la longueur du tuple."""
    return vals[-1]


def plot_cumul_errors(results, title="Erreurs cumulées — classificateurs en ligne"):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, vals in results.items():
        hist = _get_hist(vals)
        color = COLORS.get(name, "#333333")
        ls = "--" if name == "Perceptron" else "-"
        ax.plot(hist["cumul_errors"], label=name,
                color=color, linewidth=2, linestyle=ls)
    ax.set_xlabel("Tour t")
    ax.set_ylabel("Erreurs cumulées MT")
    ax.set_title(title)
    ax.legend(ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_instant_losses_online(results, title="Pertes instantanées (lissées)"):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, vals in results.items():
        hist = _get_hist(vals)
        losses = np.array(hist["instant_loss"])
        window = max(1, len(losses) // 80)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        color = COLORS.get(name, "#333333")
        ax.plot(smoothed, label=name, color=color, linewidth=2)
    ax.set_xlabel("Tour t")
    ax.set_ylabel("Perte instantanée (lissée)")
    ax.set_title(title)
    ax.legend(ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_w_norms(results, title="Évolution de ||wt||₂"):
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, vals in results.items():
        hist = _get_hist(vals)
        color = COLORS.get(name, "#333333")
        ax.plot(hist["w_norm"], label=name, color=color, linewidth=2)
    ax.set_xlabel("Tour t")
    ax.set_ylabel("||wt||₂")
    ax.set_title(title)
    ax.legend(ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_sparsity(results_reg, title="Sparsité de w (fraction de zéros) — L1 vs L2"):
    """Compare la sparsité induite par L1 vs L2."""
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#378ADD", "#D85A30", "#1D9E75", "#D4537E", "#EF9F27"]
    for i, (name, vals) in enumerate(results_reg.items()):
        hist = _get_hist(vals)
        if "sparsity" in hist:
            ax.plot(hist["sparsity"], label=name,
                    color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel("Tour t")
    ax.set_ylabel("Fraction de wi ≈ 0")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def summary_table_classifiers(results, X_test, y_test):
    """Tableau récap : erreurs cumulées, accuracy, F1, ||w||."""
    from TP2.metrics import accuracy as acc_fn, f1_score as f1_fn
    from TP2.perceptron import predict as pred_fn

    print(f"\n{'Algorithme':<20} {'Err. cum.':>10} {'Acc.':>8}"
          f" {'F1':>8} {'||w||':>8} {'Sparsité':>10}")
    print("-" * 70)
    for name, vals in results.items():
        w, b, hist = vals[0], vals[1], vals[2]
        y_pred = pred_fn(w, b, X_test)
        acc = acc_fn(y_test, y_pred)
        f1  = f1_fn(y_test, y_pred)
        mt  = hist["cumul_errors"][-1]
        wn  = hist["w_norm"][-1]
        spar = hist.get("sparsity", [0])[-1]
        print(f"{name:<20} {mt:>10d} {acc:>8.4f}"
              f" {f1:>8.4f} {wn:>8.3f} {spar:>10.3f}")