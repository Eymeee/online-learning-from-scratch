"""
kernels.py — Online Learning with Kernels (TP5)
================================================
Contenu :
  1. Fonctions noyaux : linéaire, polynomial, polynomial décalé, gaussien, sigmoïde
  2. Matrice de Gram
  3. Perceptron kernelisé
  4. OSD kernelisé (Kernelized OGD)
  5. Comparaison noyaux
  6. Visualisation
"""

import numpy as np
import matplotlib.pyplot as plt


# ===========================================================================
# 1. FONCTIONS NOYAUX
# ===========================================================================

def kernel_linear(x, z):
    """k(x,z) = x^T z"""
    return float(np.dot(x, z))


def kernel_polynomial(x, z, d=2):
    """k(x,z) = (x^T z)^d"""
    return float(np.dot(x, z) ** d)


def kernel_polynomial_shifted(x, z, d=2, c=1.0):
    """k(x,z) = (x^T z + c)^d"""
    return float((np.dot(x, z) + c) ** d)


def kernel_gaussian(x, z, sigma=1.0):
    """k(x,z) = exp(-||x-z||^2 / (2*sigma^2))"""
    diff = x - z
    return float(np.exp(-np.dot(diff, diff) / (2.0 * sigma ** 2)))


def kernel_sigmoid(x, z, kappa=1.0, c=0.0):
    """k(x,z) = tanh(kappa * x^T z + c)"""
    return float(np.tanh(kappa * np.dot(x, z) + c))


def get_kernel(name, **params):
    """
    Retourne une fonction noyau k(x, z) par son nom.

    name : "linear" | "poly" | "poly_shifted" | "gaussian" | "sigmoid"
    """
    kernels = {
        "linear":       lambda x, z: kernel_linear(x, z),
        "poly":         lambda x, z: kernel_polynomial(x, z, d=params.get("d", 2)),
        "poly_shifted": lambda x, z: kernel_polynomial_shifted(
                            x, z, d=params.get("d", 2), c=params.get("c", 1.0)),
        "gaussian":     lambda x, z: kernel_gaussian(x, z,
                            sigma=params.get("sigma", 1.0)),
        "sigmoid":      lambda x, z: kernel_sigmoid(x, z,
                            kappa=params.get("kappa", 1.0),
                            c=params.get("c", 0.0)),
    }
    if name not in kernels:
        raise ValueError(f"Noyau '{name}' inconnu. Choisir parmi {list(kernels)}")
    return kernels[name]


def gram_matrix(X, kernel_fn):
    """
    Calcule la matrice de Gram K[i,j] = k(xi, xj).

    Paramètres
    ----------
    X         : np.ndarray (n, p)
    kernel_fn : callable(x, z) → float

    Retourne
    --------
    K : np.ndarray (n, n)
    """
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val = kernel_fn(X[i], X[j])
            K[i, j] = val
            K[j, i] = val
    return K


# ===========================================================================
# 2. PERCEPTRON KERNELISÉ
# ===========================================================================

def kernelized_perceptron(X, y, kernel_fn):
    """
    Perceptron kernelisé en ligne.

    Représentation duale : ft(x) = sum_{s in S} alpha_s * y_s * k(x_s, x)
    En cas d'erreur : ft+1 = ft + yt * k(xt, ·)

    Paramètres
    ----------
    kernel_fn : callable(x, z) → float

    Retourne
    --------
    support_vectors : list of (xi, yi) — vecteurs de support
    alphas          : list of float — coefficients
    history         : dict
    """
    n = len(y)
    support_vectors = []   # (xi, yi)
    alphas = []            # coefficients (toujours 1 ici)

    history = {
        "cumul_errors": [],
        "instant_loss": [],
        "n_support":    [],
    }
    cumul_err = 0

    def predict_t(x):
        if not support_vectors:
            return 0.0
        score = sum(
            alpha * sv_y * kernel_fn(sv_x, x)
            for (sv_x, sv_y), alpha in zip(support_vectors, alphas)
        )
        return score

    for t in range(n):
        score = predict_t(X[t])
        y_hat = 1 if score >= 0 else -1
        err = int(y_hat != y[t])
        loss = max(0.0, -y[t] * score)
        cumul_err += err

        if err:
            support_vectors.append((X[t].copy(), y[t]))
            alphas.append(1.0)

        history["cumul_errors"].append(cumul_err)
        history["instant_loss"].append(loss)
        history["n_support"].append(len(support_vectors))

    return support_vectors, alphas, history


def kernelized_perceptron_predict(support_vectors, alphas, kernel_fn, X_test):
    """Prédiction sur X_test à partir de la représentation duale."""
    y_pred = []
    for x in X_test:
        if not support_vectors:
            y_pred.append(1)
            continue
        score = sum(
            alpha * sv_y * kernel_fn(sv_x, x)
            for (sv_x, sv_y), alpha in zip(support_vectors, alphas)
        )
        y_pred.append(1 if score >= 0 else -1)
    return np.array(y_pred)


# ===========================================================================
# 3. OSD KERNELISÉ
# ===========================================================================

def kernelized_osd(X, y, kernel_fn, eta0=0.1, decay="sqrt"):
    """
    Online Subgradient Descent kernelisé.

    ft+1 = ft + ηt * yt * k(xt, ·)  si perte hinge > 0

    Représentation duale :
      f(x) = sum_t eta_t * yt * k(xt, x)  [pour les tours où ℓt > 0]

    Retourne
    --------
    support_vectors : list of (xi, yi, eta_t)
    history         : dict
    """
    n = len(y)
    support_vectors = []   # (xi, yi, eta_t)

    history = {
        "cumul_errors": [],
        "instant_loss": [],
        "n_support":    [],
    }
    cumul_err = 0

    def score_t(x):
        return sum(
            eta * sv_y * kernel_fn(sv_x, x)
            for sv_x, sv_y, eta in support_vectors
        )

    for t in range(n):
        if decay == "sqrt":
            step = eta0 / np.sqrt(t + 1)
        elif decay == "poly":
            step = eta0 / (t + 1)
        else:
            step = eta0

        sc = score_t(X[t])
        y_hat = 1 if sc >= 0 else -1
        err = int(y_hat != y[t])
        loss_t = max(0.0, 1.0 - y[t] * sc)
        cumul_err += err

        if loss_t > 0:
            support_vectors.append((X[t].copy(), y[t], step))

        history["cumul_errors"].append(cumul_err)
        history["instant_loss"].append(loss_t)
        history["n_support"].append(len(support_vectors))

    return support_vectors, history


def kernelized_osd_predict(support_vectors, kernel_fn, X_test):
    """Prédiction OSD kernelisé."""
    y_pred = []
    for x in X_test:
        sc = sum(
            eta * sv_y * kernel_fn(sv_x, x)
            for sv_x, sv_y, eta in support_vectors
        )
        y_pred.append(1 if sc >= 0 else -1)
    return np.array(y_pred)


# ===========================================================================
# 4. COMPARAISON DE NOYAUX
# ===========================================================================

def compare_kernels(X_train, y_train, X_test, y_test,
                    kernel_configs=None, method="perceptron"):
    """
    Compare plusieurs noyaux sur un problème de classification.

    kernel_configs : list of dict avec clés 'name' et params optionnels
    method         : "perceptron" | "osd"

    Retourne
    --------
    results : dict {label: {"history": hist, "accuracy": float,
                             "n_support": int}}
    """
    from TP2.metrics import accuracy as acc_fn

    if kernel_configs is None:
        kernel_configs = [
            {"name": "linear"},
            {"name": "poly",         "d": 2},
            {"name": "poly",         "d": 3},
            {"name": "poly_shifted", "d": 2},
            {"name": "gaussian",     "sigma": 1.0},
        ]

    results = {}
    for cfg in kernel_configs:
        name = cfg.pop("name")
        label = name + ("(" + ",".join(f"{k}={v}" for k,v in cfg.items()) + ")"
                         if cfg else "")
        kfn = get_kernel(name, **cfg)
        cfg["name"] = name   # remettre pour réutilisation

        if method == "perceptron":
            sv, alphas, hist = kernelized_perceptron(X_train, y_train, kfn)
            y_pred = kernelized_perceptron_predict(sv, alphas, kfn, X_test)
            n_sv = len(sv)
        else:
            sv, hist = kernelized_osd(X_train, y_train, kfn)
            y_pred = kernelized_osd_predict(sv, kfn, X_test)
            n_sv = len(sv)

        acc = float(np.mean(y_pred == y_test))
        results[label] = {"history": hist, "accuracy": acc, "n_support": n_sv}

    return results


# ===========================================================================
# 5. VISUALISATION
# ===========================================================================

COLORS = ["#378ADD", "#D85A30", "#1D9E75", "#D4537E", "#EF9F27", "#888780"]


def plot_kernel_comparison(results, title="Comparaison des noyaux"):
    """Tableau visuel des performances."""
    labels = list(results.keys())
    accs   = [results[k]["accuracy"]  for k in labels]
    n_svs  = [results[k]["n_support"] for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(labels))

    bars = axes[0].bar(x, accs, color=COLORS[:len(labels)])
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Accuracy test")
    axes[0].set_title("Accuracy par noyau")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.3, axis="y")
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, acc + 0.01,
                     f"{acc:.3f}", ha="center", fontsize=9)

    bars2 = axes[1].bar(x, n_svs, color=COLORS[:len(labels)])
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("Nombre de vecteurs de support")
    axes[1].set_title("Complexité du modèle (# supports)")
    axes[1].grid(True, linestyle="--", alpha=0.3, axis="y")

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig


def plot_cumul_errors_kernels(results, title="Erreurs cumulées par noyau"):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, res) in enumerate(results.items()):
        hist = res["history"]
        ax.plot(hist["cumul_errors"], label=label,
                color=COLORS[i % len(COLORS)], linewidth=2)
    ax.set_xlabel("Tour t"); ax.set_ylabel("Erreurs cumulées")
    ax.set_title(title); ax.legend(ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_support_growth(results, title="Croissance des vecteurs de support"):
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (label, res) in enumerate(results.items()):
        hist = res["history"]
        ax.plot(hist["n_support"], label=label,
                color=COLORS[i % len(COLORS)], linewidth=2)
    ax.set_xlabel("Tour t"); ax.set_ylabel("# vecteurs de support")
    ax.set_title(title); ax.legend(ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_decision_boundary_kernel(sv, alphas, kernel_fn,
                                   X, y, title="Frontière de décision (noyau)"):
    """Trace la frontière de décision kernelisée en 2D."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, color in [(-1, "#378ADD"), (1, "#D85A30")]:
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], c=color, alpha=0.6,
                   s=20, edgecolors="none", label=f"Classe {label}")

    x_min, x_max = X[:, 0].min()-0.5, X[:, 0].max()+0.5
    y_min, y_max = X[:, 1].min()-0.5, X[:, 1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120),
                          np.linspace(y_min, y_max, 120))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = kernelized_perceptron_predict(sv, alphas, kernel_fn, grid)
    zz = zz.reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.12, colors=["#378ADD", "#D85A30"])
    ax.contour(xx, yy, zz, colors="black", linewidths=1)
    ax.set_title(title); ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig