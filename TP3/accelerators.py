"""
accelerators.py — Accélérateurs pour gradient et sous-gradient (TP3)
=====================================================================
Contenu :
  1. Momentum (Heavy Ball)
  2. Nesterov Accelerated Gradient (NAG)
  3. AdaGrad
  4. RMSProp
  5. Adam
  6. Aitken (optionnel, sur suite scalaire)
  7. Interface unifiée : run_accelerator(), run_all()
  8. Comparaison et visualisation
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath('..'))


# ===========================================================================
# 1. MOMENTUM (Heavy Ball)
# ===========================================================================

def momentum(grad_fn, loss_fn, theta0, alpha=0.01, beta=0.9,
             n_iter=1000, project_fn=None):
    """
    v_{t+1} = beta * v_t + grad(theta_t)
    theta_{t+1} = theta_t - alpha * v_{t+1}
    """
    theta = theta0.copy()
    v = np.zeros_like(theta)
    history = {"cost": [], "grad_norm": []}

    for _ in range(n_iter):
        g = grad_fn(theta)
        v = beta * v + g
        theta = theta - alpha * v
        if project_fn is not None:
            theta = project_fn(theta)
        history["cost"].append(loss_fn(theta))
        history["grad_norm"].append(float(np.linalg.norm(g)))

    return theta, history


# ===========================================================================
# 2. NESTEROV ACCELERATED GRADIENT (NAG)
# ===========================================================================

def nesterov(grad_fn, loss_fn, theta0, alpha=0.01, beta=0.9,
             n_iter=1000, project_fn=None):
    """
    theta_look  = theta_t - alpha * beta * v_t
    v_{t+1}     = beta * v_t + grad(theta_look)
    theta_{t+1} = theta_t - alpha * v_{t+1}
    """
    theta = theta0.copy()
    v = np.zeros_like(theta)
    history = {"cost": [], "grad_norm": []}

    for _ in range(n_iter):
        theta_look = theta - alpha * beta * v
        g = grad_fn(theta_look)
        v = beta * v + g
        theta = theta - alpha * v
        if project_fn is not None:
            theta = project_fn(theta)
        history["cost"].append(loss_fn(theta))
        history["grad_norm"].append(float(np.linalg.norm(g)))

    return theta, history


# ===========================================================================
# 3. ADAGRAD
# ===========================================================================

def adagrad(grad_fn, loss_fn, theta0, alpha=0.1, eps=1e-8,
            n_iter=1000, project_fn=None):
    """
    G_{t,j}       = G_{t-1,j} + g_{t,j}^2
    theta_{t+1,j} = theta_{t,j} - alpha/sqrt(G_{t,j} + eps) * g_{t,j}
    """
    theta = theta0.copy()
    G = np.zeros_like(theta)
    history = {"cost": [], "grad_norm": [], "eff_lr": []}

    for _ in range(n_iter):
        g = grad_fn(theta)
        G += g ** 2
        lr = alpha / (np.sqrt(G) + eps)
        theta = theta - lr * g
        if project_fn is not None:
            theta = project_fn(theta)
        history["cost"].append(loss_fn(theta))
        history["grad_norm"].append(float(np.linalg.norm(g)))
        history["eff_lr"].append(float(np.mean(lr)))

    return theta, history


# ===========================================================================
# 4. RMSPROP
# ===========================================================================

def rmsprop(grad_fn, loss_fn, theta0, alpha=0.01, rho=0.9, eps=1e-8,
            n_iter=1000, project_fn=None):
    """
    s_t         = rho * s_{t-1} + (1 - rho) * g_t^2
    theta_{t+1} = theta_t - alpha/sqrt(s_t + eps) * g_t
    """
    theta = theta0.copy()
    s = np.zeros_like(theta)
    history = {"cost": [], "grad_norm": [], "eff_lr": []}

    for _ in range(n_iter):
        g = grad_fn(theta)
        s = rho * s + (1.0 - rho) * g ** 2
        lr = alpha / (np.sqrt(s) + eps)
        theta = theta - lr * g
        if project_fn is not None:
            theta = project_fn(theta)
        history["cost"].append(loss_fn(theta))
        history["grad_norm"].append(float(np.linalg.norm(g)))
        history["eff_lr"].append(float(np.mean(lr)))

    return theta, history


# ===========================================================================
# 5. ADAM
# ===========================================================================

def adam(grad_fn, loss_fn, theta0, alpha=0.001, beta1=0.9, beta2=0.999,
         eps=1e-8, n_iter=1000, project_fn=None):
    """
    m_t         = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t         = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat       = m_t / (1 - beta1^t)
    v_hat       = v_t / (1 - beta2^t)
    theta_{t+1} = theta_t - alpha * m_hat / (sqrt(v_hat) + eps)
    """
    theta = theta0.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history = {"cost": [], "grad_norm": [], "eff_lr": []}

    for t in range(1, n_iter + 1):
        g = grad_fn(theta)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g ** 2
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        lr = alpha / (np.sqrt(v_hat) + eps)
        theta = theta - lr * m_hat
        if project_fn is not None:
            theta = project_fn(theta)
        history["cost"].append(loss_fn(theta))
        history["grad_norm"].append(float(np.linalg.norm(g)))
        history["eff_lr"].append(float(np.mean(lr)))

    return theta, history


# ===========================================================================
# 6. AITKEN (optionnel)
# ===========================================================================

def aitken_acceleration(sequence):
    """
    Accélération d'Aitken sur une suite scalaire convergente.

    u_t^(A) = u_t - (u_{t+1} - u_t)^2 / (u_{t+2} - 2*u_{t+1} + u_t)

    Retourne la suite accélérée de longueur len(sequence) - 2.
    """
    u = np.array(sequence, dtype=float)
    if len(u) < 3:
        raise ValueError("Au moins 3 éléments nécessaires.")
    denom = u[2:] - 2 * u[1:-1] + u[:-2]
    safe  = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return u[:-2] - (u[1:-1] - u[:-2]) ** 2 / safe


def plot_aitken(costs, title="Accélération d'Aitken"):
    acc = aitken_acceleration(costs)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(costs, color="#378ADD", linewidth=2, label="Suite originale")
    ax.plot(range(len(acc)), acc, color="#D85A30", linewidth=2,
            linestyle="--", label="Aitken")
    ax.set_xlabel("Itération"); ax.set_ylabel("Coût")
    ax.set_title(title); ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


# ===========================================================================
# 7. INTERFACE UNIFIÉE
# ===========================================================================

_REGISTRY = {
    "Standard": None,
    "Momentum": momentum,
    "Nesterov": nesterov,
    "AdaGrad": adagrad,
    "RMSProp": rmsprop,
    "Adam": adam,
}

_NAME_ALIASES = {"Gradient standard": "Standard"}


def gradient_standard(grad_fn, loss_fn, theta0, alpha=0.01,
                      n_iter=1000, project_fn=None):
    """Descente de gradient « vanilla » (alias de run_accelerator Standard)."""
    return run_accelerator(
        "Standard", grad_fn, loss_fn, theta0,
        n_iter=n_iter, project_fn=project_fn, alpha=alpha,
    )


def run_accelerator(name, grad_fn, loss_fn, theta0,
                    n_iter=1000, project_fn=None, **kwargs):
    """
    Lance un accélérateur par son nom.

    name     : str — clé dans _REGISTRY
    **kwargs : hyperparamètres de l'accélérateur
    """
    name = _NAME_ALIASES.get(name, name)
    if name == "Standard":
        alpha = kwargs.get("alpha", 0.01)
        theta = theta0.copy()
        history = {"cost": [], "grad_norm": []}
        for _ in range(n_iter):
            g = grad_fn(theta)
            theta = theta - alpha * g
            if project_fn is not None:
                theta = project_fn(theta)
            history["cost"].append(loss_fn(theta))
            history["grad_norm"].append(float(np.linalg.norm(g)))
        return theta, history

    fn = _REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"Inconnu : {name}. Disponibles : {list(_REGISTRY)}")
    return fn(grad_fn, loss_fn, theta0.copy(),
              n_iter=n_iter, project_fn=project_fn, **kwargs)


def run_all(grad_fn, loss_fn, theta0, configs=None, n_iter=1000,
            project_fn=None, **kwargs):
    """
    Lance plusieurs accélérateurs d'un coup.

    Mode 1 — dict explicite : configs={nom: {hyperparams...}, ...}

    Mode 2 — notebook TP3 : run_all(..., n_iter=800,
              alpha_std=..., alpha_mom=..., beta_mom=..., ...)

    Retourne : dict {nom: (theta, history)}
    """
    if configs is None:
        keys_tp3 = (
            "alpha_std", "alpha_mom", "beta_mom", "alpha_nag", "beta_nag",
            "alpha_ada", "alpha_rms", "rho_rms", "alpha_adam",
        )
        missing = [k for k in keys_tp3 if k not in kwargs]
        if missing:
            raise TypeError(
                "run_all : passer configs=dict(...) ou tous les mots-clés "
                f"alpha_std, alpha_mom, ... Manquants : {missing}"
            )
        configs = {
            "Standard": {"alpha": kwargs["alpha_std"]},
            "Momentum": {"alpha": kwargs["alpha_mom"], "beta": kwargs["beta_mom"]},
            "Nesterov": {"alpha": kwargs["alpha_nag"], "beta": kwargs["beta_nag"]},
            "AdaGrad": {"alpha": kwargs["alpha_ada"]},
            "RMSProp": {"alpha": kwargs["alpha_rms"], "rho": kwargs["rho_rms"]},
            "Adam": {"alpha": kwargs["alpha_adam"]},
        }
    results = {}
    for name, kw in configs.items():
        theta, history = run_accelerator(
            name, grad_fn, loss_fn, theta0.copy(),
            n_iter=n_iter, project_fn=project_fn, **kw
        )
        results[name] = (theta, history)
    return results


# ===========================================================================
# 8. COMPARAISON ET VISUALISATION
# ===========================================================================

_COLORS = ["#888780", "#378ADD", "#D85A30",
           "#1D9E75", "#D4537E", "#EF9F27", "#7F77DD"]


def plot_cost_comparison(results, title="Accélérateurs — Coût", log=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, (_, hist)) in enumerate(results.items()):
        fn = ax.semilogy if log else ax.plot
        fn(hist["cost"], label=name, linewidth=2,
           color=_COLORS[i % len(_COLORS)])
    ax.set_xlabel("Itération")
    ax.set_ylabel("Coût" + (" (log)" if log else ""))
    ax.set_title(title); ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_grad_norm_comparison(results, title="Accélérateurs — Norme gradient"):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, (_, hist)) in enumerate(results.items()):
        ax.semilogy(hist["grad_norm"], label=name, linewidth=2,
                    color=_COLORS[i % len(_COLORS)])
    ax.set_xlabel("Itération"); ax.set_ylabel("||g|| (log)")
    ax.set_title(title); ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_comparison(results, metric="cost", title=None, log=False):
    """Wrapper notebook : metric='cost' | 'grad_norm'."""
    if metric == "cost":
        t = title if title is not None else "Accélérateurs — Coût"
        return plot_cost_comparison(results, title=t, log=log)
    if metric == "grad_norm":
        t = title if title is not None else "Accélérateurs — Norme gradient"
        return plot_grad_norm_comparison(results, title=t)
    raise ValueError("metric doit être 'cost' ou 'grad_norm'")


def plot_comparison_grid(results, log_cost=False):
    """Deux panneaux : coût et norme de gradient."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (name, (_, hist)) in enumerate(results.items()):
        fn = axes[0].semilogy if log_cost else axes[0].plot
        fn(hist["cost"], label=name, linewidth=2, color=_COLORS[i % len(_COLORS)])
        axes[1].semilogy(hist["grad_norm"], label=name, linewidth=2,
                         color=_COLORS[i % len(_COLORS)])
    axes[0].set_xlabel("Itération"); axes[0].set_ylabel("Coût")
    axes[0].set_title("Coût"); axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[1].set_xlabel("Itération"); axes[1].set_ylabel("||g|| (log)")
    axes[1].set_title("Norme du gradient"); axes[1].legend()
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_beta_sensitivity(grad_fn, loss_fn, theta0, method="momentum",
                          betas=(0.5, 0.8, 0.9, 0.95), alpha=0.01,
                          n_iter=1000, project_fn=None):
    """Sensibilité de Momentum ou Nesterov au paramètre β."""
    method = method.lower()
    if method == "momentum":
        acc_name = "Momentum"
    elif method == "nesterov":
        acc_name = "Nesterov"
    else:
        raise ValueError(f"méthode inconnue : {method}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, beta in enumerate(betas):
        _, hist = run_accelerator(
            acc_name, grad_fn, loss_fn, theta0.copy(),
            n_iter=n_iter, project_fn=project_fn, alpha=alpha, beta=beta,
        )
        ax.plot(hist["cost"], label=f"β={beta}", linewidth=2,
                color=_COLORS[i % len(_COLORS)])
    ax.set_xlabel("Itération"); ax.set_ylabel("Coût")
    ax.set_title(f"Sensibilité de {acc_name} à β")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_rho_sensitivity(grad_fn, loss_fn, theta0, rhos=(0.9, 0.95, 0.99),
                         alpha=0.01, n_iter=1000, project_fn=None):
    """Sensibilité de RMSProp au paramètre ρ."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, rho in enumerate(rhos):
        _, hist = run_accelerator(
            "RMSProp", grad_fn, loss_fn, theta0.copy(),
            n_iter=n_iter, project_fn=project_fn, alpha=alpha, rho=rho,
        )
        ax.plot(hist["cost"], label=f"ρ={rho}", linewidth=2,
                color=_COLORS[i % len(_COLORS)])
    ax.set_xlabel("Itération"); ax.set_ylabel("Coût")
    ax.set_title("Sensibilité de RMSProp à ρ")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_effective_lr(results, title="Pas effectif moyen"):
    fig, ax = plt.subplots(figsize=(9, 4))
    i = 0
    for name, (_, hist) in results.items():
        if "eff_lr" in hist:
            ax.plot(hist["eff_lr"], label=name, linewidth=2,
                    color=_COLORS[(i + 2) % len(_COLORS)])
            i += 1
    ax.set_xlabel("Itération"); ax.set_ylabel("Pas effectif moyen")
    ax.set_title(title); ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def summary_table(results):
    """Affiche coût final, coût min et norme grad finale."""
    print(f"\n{'Accélérateur':<22} {'Coût final':>12}"
          f" {'Coût min':>12} {'||g|| final':>14}")
    print("-" * 62)
    for name, (_, hist) in results.items():
        print(f"{name:<22}"
              f" {hist['cost'][-1]:>12.6f}"
              f" {min(hist['cost']):>12.6f}"
              f" {hist['grad_norm'][-1]:>14.6f}")


def sensitivity_analysis(grad_fn, loss_fn, theta0, acc_name,
                          param_name, param_values,
                          n_iter=500, fixed_kwargs=None):
    """
    Sensibilité à un hyperparamètre donné.

    Retourne (results_dict, fig)
    results_dict : {param_value: history}
    """
    fixed_kwargs = fixed_kwargs or {}
    results = {}
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, val in enumerate(param_values):
        kw = {**fixed_kwargs, param_name: val}
        _, hist = run_accelerator(acc_name, grad_fn, loss_fn,
                                   theta0.copy(), n_iter=n_iter, **kw)
        results[val] = hist
        ax.plot(hist["cost"], label=f"{param_name}={val}",
                linewidth=2, color=_COLORS[i % len(_COLORS)])

    ax.set_xlabel("Itération"); ax.set_ylabel("Coût")
    ax.set_title(f"Sensibilité de {acc_name} à {param_name}")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return results, fig