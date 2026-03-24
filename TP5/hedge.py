"""
hedge.py — Prediction with Expert Advice & algorithme Hedge (TP5)
=================================================================
Contenu :
  1. Algorithme Hedge
  2. Regret par rapport au meilleur expert
  3. Exemples synthétiques (aléatoire, experts changeants, classification)
  4. Visualisation
"""

import numpy as np
import matplotlib.pyplot as plt


# ===========================================================================
# 1. ALGORITHME HEDGE
# ===========================================================================

def hedge(losses_matrix, beta=0.5):
    """
    Algorithme Hedge.

    pt,i = wt,i / sum_j wt,j
    Lt_learner = pt · lt
    wt+1,i = wt,i * beta^(lt,i)

    Paramètres
    ----------
    losses_matrix : np.ndarray (T, N) — perte de l'expert i au tour t
    beta          : float ∈ (0,1]

    Retourne
    --------
    history : dict avec learner_losses, expert_losses, weights,
              distributions, regret, cumul_learner, best_fixed_loss
    """
    T, N = losses_matrix.shape
    w = np.ones(N)
    learner_losses, cumul_learner_list = [], []
    expert_losses   = np.zeros((T, N))
    weights_history = np.zeros((T, N))
    dist_history    = np.zeros((T, N))
    cumul_learner   = 0.0
    cumul_experts   = np.zeros(N)

    for t in range(T):
        p = w / w.sum()
        lt = losses_matrix[t]
        ll = float(np.dot(p, lt))
        cumul_learner  += ll
        cumul_experts  += lt
        w = w * (beta ** lt)
        learner_losses.append(ll)
        cumul_learner_list.append(cumul_learner)
        expert_losses[t]   = cumul_experts.copy()
        weights_history[t] = w.copy()
        dist_history[t]    = p.copy()

    cumul_arr = np.array(cumul_learner_list)
    regret = cumul_arr - np.array(
        [np.min(expert_losses[t]) for t in range(T)]
    )

    return {
        "learner_losses":  learner_losses,
        "expert_losses":   expert_losses,
        "weights":         weights_history,
        "distributions":   dist_history,
        "regret":          regret,
        "cumul_learner":   cumul_arr,
        "best_fixed_loss": float(np.min(expert_losses[-1])),
    }


def study_beta_effect(losses_matrix, betas=None):
    if betas is None:
        betas = [0.1, 0.5, 0.8, 0.9, 0.99]
    return {f"β={b}": hedge(losses_matrix, beta=b) for b in betas}


# ===========================================================================
# 2. EXEMPLES SYNTHÉTIQUES
# ===========================================================================

def make_expert_losses_random(T=500, N=3, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=(T, N))


def make_expert_losses_shifting(T=500, N=3, seed=42):
    """Meilleur expert change à mi-parcours."""
    rng = np.random.default_rng(seed)
    losses = rng.uniform(0.3, 0.7, size=(T, N))
    switch = T // 2
    losses[:switch, 0] *= 0.2
    losses[switch:, 1] *= 0.2
    losses[:switch, 1] *= 1.5
    return np.clip(losses, 0, 1)


def make_expert_losses_classification(X, y, classifiers):
    """
    Pertes 0/1 de N classifieurs (experts) sur le flux (X, y).

    classifiers : list of (w, b)
    """
    T, N = len(y), len(classifiers)
    losses = np.zeros((T, N))
    for i, (w, b) in enumerate(classifiers):
        scores = X @ w + b
        y_hat = np.where(scores >= 0, 1, -1)
        losses[:, i] = (y_hat != y).astype(float)
    return losses


# ===========================================================================
# 3. VISUALISATION
# ===========================================================================

EC = ["#378ADD", "#D85A30", "#1D9E75", "#D4537E"]


def plot_hedge_regret(hist, title="Regret cumulé — Hedge"):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hist["regret"], color="#378ADD", linewidth=2, label="Regret")
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Tour t"); ax.set_ylabel("Regret cumulé")
    ax.set_title(title); ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_learner_vs_experts(hist, n_experts=None,
                             title="Apprenant vs experts — Pertes cumulées"):
    N = min(hist["expert_losses"].shape[1], n_experts or 999)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist["cumul_learner"], color="black", linewidth=2.5,
            label="Apprenant (Hedge)")
    for i in range(N):
        ax.plot(hist["expert_losses"][:, i], color=EC[i % len(EC)],
                linewidth=1.5, linestyle="--", alpha=0.8,
                label=f"Expert {i+1}")
    ax.set_xlabel("Tour t"); ax.set_ylabel("Perte cumulée")
    ax.set_title(title); ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_weights_evolution(hist, n_experts=None,
                            title="Évolution des poids — Hedge"):
    N = min(hist["weights"].shape[1], n_experts or 999)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(N):
        w_rel = hist["weights"][:, i] / (hist["weights"].sum(axis=1) + 1e-12)
        ax.plot(w_rel, color=EC[i % len(EC)],
                linewidth=2, label=f"Expert {i+1}")
    ax.set_xlabel("Tour t"); ax.set_ylabel("Poids relatif pt,i")
    ax.set_title(title); ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_beta_comparison(results_beta, title="Effet de β — Hedge"):
    colors = ["#378ADD", "#D85A30", "#1D9E75", "#D4537E", "#888780"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for i, (label, hist) in enumerate(results_beta.items()):
        c = colors[i % len(colors)]
        axes[0].plot(hist["regret"],        label=label, color=c, linewidth=2)
        axes[1].plot(hist["cumul_learner"], label=label, color=c, linewidth=2)
    for ax, ttl, yl in zip(axes,
        ["Regret cumulé selon β", "Perte cumulée selon β"],
        ["Regret", "Perte cumulée"]):
        ax.set_title(ttl); ax.set_xlabel("Tour t"); ax.set_ylabel(yl)
        ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].axhline(0, color="black", linestyle=":", linewidth=1)
    plt.suptitle(title, y=1.01)
    plt.tight_layout()
    return fig


def hedge_summary(hist, N):
    print(f"{'='*45}")
    print(f"  Résumé — Hedge ({N} experts)")
    print(f"{'='*45}")
    print(f"  Perte cumulée apprenant : {hist['cumul_learner'][-1]:.4f}")
    print(f"  Perte meilleur expert   : {hist['best_fixed_loss']:.4f}")
    print(f"  Regret final            : {hist['regret'][-1]:.4f}")
    best = int(np.argmin(hist['expert_losses'][-1]))
    print(f"  Meilleur expert         : Expert {best+1}")
    print(f"{'='*45}")