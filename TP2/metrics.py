"""
metrics.py — Métriques de classification binaire (TP2)
=======================================================
Contenu :
  1. Accuracy, précision, rappel, F1-score
  2. Matrice de confusion
  3. Rapport complet
  4. Courbes biais-variance (train/test)
  5. Visualisations
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath('..'))


# ===========================================================================
# 1. MÉTRIQUES DE BASE
# ===========================================================================

def accuracy(y_true, y_pred):
    """Taux de bonne classification."""
    return float(np.mean(y_true == y_pred))


def confusion_matrix_binary(y_true, y_pred, pos_label=1):
    """
    Matrice de confusion 2x2 pour classification binaire.

    Retourne
    --------
    tp, fp, tn, fn : int
    """
    tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    fp = int(np.sum((y_true != pos_label) & (y_pred == pos_label)))
    tn = int(np.sum((y_true != pos_label) & (y_pred != pos_label)))
    fn = int(np.sum((y_true == pos_label) & (y_pred != pos_label)))
    return tp, fp, tn, fn


def precision(y_true, y_pred, pos_label=1):
    """Précision = TP / (TP + FP)"""
    tp, fp, _, _ = confusion_matrix_binary(y_true, y_pred, pos_label)
    return tp / (tp + fp + 1e-12)


def recall(y_true, y_pred, pos_label=1):
    """Rappel = TP / (TP + FN)"""
    tp, _, _, fn = confusion_matrix_binary(y_true, y_pred, pos_label)
    return tp / (tp + fn + 1e-12)


def f1_score(y_true, y_pred, pos_label=1):
    """F1 = 2 * précision * rappel / (précision + rappel)"""
    p = precision(y_true, y_pred, pos_label)
    r = recall(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r + 1e-12)


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Matrice de confusion générale (multi-classes possible).

    Retourne
    --------
    cm     : np.ndarray shape (n_classes, n_classes)
    labels : list
    """
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt], label_to_idx[yp]] += 1
    return cm, labels


# ===========================================================================
# 2. RAPPORT COMPLET
# ===========================================================================

def classification_report(y_true, y_pred, pos_label=1, name="Modèle"):
    """
    Affiche et retourne toutes les métriques.

    Retourne
    --------
    metrics : dict
    """
    acc   = accuracy(y_true, y_pred)
    prec  = precision(y_true, y_pred, pos_label)
    rec   = recall(y_true, y_pred, pos_label)
    f1    = f1_score(y_true, y_pred, pos_label)
    tp, fp, tn, fn = confusion_matrix_binary(y_true, y_pred, pos_label)
    cm, labels = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*45}")
    print(f"  Rapport de classification — {name}")
    print(f"{'='*45}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Précision : {prec:.4f}")
    print(f"  Rappel    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"{'='*45}\n")

    return {
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "cm": cm, "labels": labels
    }


def compare_models(models_metrics, metric="accuracy"):
    """
    Affiche un tableau comparatif de plusieurs modèles.

    Paramètres
    ----------
    models_metrics : dict {nom_modele: metrics_dict}
    metric         : métrique principale à afficher en premier
    """
    keys = ["accuracy", "precision", "recall", "f1"]
    header = f"{'Modèle':<25}" + "".join(f"{k:>12}" for k in keys)
    print(header)
    print("-" * (25 + 12 * len(keys)))
    for name, m in models_metrics.items():
        row = f"{name:<25}"
        for k in keys:
            row += f"{m[k]:>12.4f}"
        print(row)


# ===========================================================================
# 3. COURBES BIAIS-VARIANCE
# ===========================================================================

def bias_variance_curve(X_train, y_train, X_test, y_test,
                         degrees, train_fn, predict_fn):
    """
    Calcule accuracy train et test pour plusieurs degrés de transformation.

    Paramètres
    ----------
    degrees    : list[int]
    train_fn   : callable(X_tr, y_tr, d) → modèle (w, b)
    predict_fn : callable(model, X) → y_pred

    Retourne
    --------
    train_accs, test_accs : lists
    """
    train_accs, test_accs = [], []
    for d in degrees:
        model = train_fn(X_train, y_train, d)
        y_tr_pred = predict_fn(model, X_train, d)
        y_te_pred = predict_fn(model, X_test,  d)
        train_accs.append(accuracy(y_train, y_tr_pred))
        test_accs.append(accuracy(y_test,  y_te_pred))
    return train_accs, test_accs


# ===========================================================================
# 4. VISUALISATIONS
# ===========================================================================

def plot_confusion_matrix(cm, labels, title="Matrice de confusion",
                           normalize=False):
    """
    Trace la matrice de confusion avec annotations.

    Paramètres
    ----------
    normalize : bool — normaliser par ligne (true positives rate)
    """
    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_plot, cmap="Blues", vmin=0)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([str(l) for l in labels])
    ax.set_yticklabels([str(l) for l in labels])
    ax.set_xlabel("Prédit", fontsize=12)
    ax.set_ylabel("Réel",   fontsize=12)
    ax.set_title(title, fontsize=13)

    thresh = cm_plot.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = f"{cm_plot[i,j]:{fmt}}"
            color = "white" if cm_plot[i, j] > thresh else "black"
            ax.text(j, i, val, ha="center", va="center",
                    color=color, fontsize=12)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def plot_bias_variance(degrees, train_accs, test_accs,
                        title="Biais-Variance — accuracy"):
    """Courbe accuracy train/test en fonction du degré."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(degrees, train_accs, marker="o", linewidth=2,
            color="#378ADD", label="Train accuracy")
    ax.plot(degrees, test_accs,  marker="s", linewidth=2,
            color="#D85A30", label="Test accuracy")

    best_d = degrees[int(np.argmax(test_accs))]
    ax.axvline(best_d, linestyle=":", color="#1D9E75",
               label=f"d* = {best_d}")

    ax.set_xlabel("Degré d")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_metric_vs_lambda(lambdas, train_metrics, val_metrics,
                           metric_name="accuracy",
                           title="Effet de λ (régularisation Ridge)"):
    """Trace une métrique train/val en fonction de λ."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(lambdas, train_metrics, marker="o", linewidth=2,
                color="#378ADD", label=f"Train {metric_name}")
    ax.semilogx(lambdas, val_metrics,   marker="s", linewidth=2,
                color="#D85A30", label=f"Val {metric_name}")

    best_lam = lambdas[int(np.argmax(val_metrics))]
    ax.axvline(best_lam, linestyle=":", color="#1D9E75",
               label=f"λ* = {best_lam:.0e}")

    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_multi_histories(results_dict, metric="cost",
                          ylabel=None, title="Comparaison"):
    """
    Trace une métrique pour plusieurs expériences.

    results_dict : dict {nom: history_dict}
    """
    colors = ["#378ADD", "#D85A30", "#1D9E75",
              "#D4537E", "#EF9F27", "#7F77DD"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, hist) in enumerate(results_dict.items()):
        if metric in hist:
            ax.plot(hist[metric], label=name, linewidth=2,
                    color=colors[i % len(colors)])
    ax.set_xlabel("Itération")
    ax.set_ylabel(ylabel or metric.capitalize())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_convergence(history, title="Convergence"):
    """Trace coût, norme du gradient et accuracy sur 3 sous-graphes."""
    has_acc = "accuracy" in history and len(history["accuracy"]) > 0
    ncols = 3 if has_acc else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    axes[0].plot(history["cost"], color="#378ADD", linewidth=2)
    axes[0].set_title("Coût J(w,b)")
    axes[0].set_xlabel("Itération")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(history["grad_norm"], color="#D85A30", linewidth=2)
    axes[1].set_title("Norme du sous-gradient")
    axes[1].set_xlabel("Itération")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    if has_acc:
        axes[2].plot(history["accuracy"], color="#1D9E75", linewidth=2)
        axes[2].set_title("Accuracy")
        axes[2].set_xlabel("Itération")
        axes[2].set_ylim(0, 1.05)
        axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(title, y=1.01)
    plt.tight_layout()
    return fig


def summary_table(results_dict, X_test, y_test, predict_fn):
    """
    Construit un tableau récapitulatif pour plusieurs modèles.

    results_dict : dict {nom: (w, b)}
    predict_fn   : callable(w, b, X) → y_pred
    """
    print(f"\n{'Modèle':<30} {'Accuracy':>10} {'Précision':>10}"
          f" {'Rappel':>10} {'F1':>10}")
    print("-" * 72)
    all_metrics = {}
    for name, (w, b) in results_dict.items():
        y_pred = predict_fn(w, b, X_test)
        acc  = accuracy(y_test, y_pred)
        prec = precision(y_test, y_pred)
        rec  = recall(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        print(f"{name:<30} {acc:>10.4f} {prec:>10.4f}"
              f" {rec:>10.4f} {f1:>10.4f}")
        all_metrics[name] = {"accuracy": acc, "precision": prec,
                              "recall": rec, "f1": f1}
    return all_metrics