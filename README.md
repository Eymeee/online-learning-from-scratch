# TPs ML — Apprentissage supervisé et en ligne

Série de 5 TPs couvrant la régression polynomiale, la classification binaire,
les accélérateurs de gradient, les méthodes online/stochastiques et
l'apprentissage en ligne supervisé.

---

## Structure du projet

```
tps_ml/
├── utils.py                      # Briques communes (partagé par tous les TPs)
├── requirements.txt
├── README.md
│
├── data/
│   └── load_datasets.py          # Chargement California Housing, Breast Cancer, Adult
│
├── TP1/****
│   ├── TP1_regression.ipynb      # Notebook principal
│   ├── polynomial.py             # Modèle poly., MSE, gradient analytique
│   └── gradient.py               # Gradient numérique (droite/gauche/centré)
│
├── TP2/
│   ├── TP2_classification.ipynb
│   ├── perceptron.py             # Classifieur linéaire, sous-gradient, descente
│   └── metrics.py                # Accuracy, F1, matrice de confusion
│
├── TP3/
│   ├── TP3_accelerators.ipynb
│   └── accelerators.py           # Momentum, Nesterov, AdaGrad, RMSProp, Adam, Aitken
│
├── TP4/
│   ├── TP4_online_stochastic.ipynb
│   ├── online.py                 # Gradient/sous-gradient en ligne, regret
│   └── stochastic.py             # SGD et SSGD, mini-lots, ordre fixe/aléatoire
│
├── TP5/
│   ├── TP5_online_supervised.ipynb
│   ├── online_classifiers.py     # Perceptron, PA, OSD, régularisation en ligne
│   ├── hedge.py                  # Prediction with Expert Advice
│   └── kernels.py                # Noyaux linéaire, polynomial, gaussien
│
└── outputs/
    ├── figures/                  # Plots exportés (.png)
    └── tables/                   # Tableaux comparatifs (.csv)
```

---

## Datasets utilisés

| TP | Dataset | Taille | Tâche |
|----|---------|--------|-------|
| TP1, TP3, TP4 | California Housing (`sklearn`) | 20 640 | Régression |
| TP2, TP3, TP4, TP5 | Breast Cancer Wisconsin (`sklearn`) | 569 | Classification binaire |
| TP5 (option) | Adult (`openml`) | 48 842 | Classification binaire |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Lancement

Ouvrir les notebooks dans l'ordre depuis le dossier de chaque TP :

```bash
cd TP1 && jupyter notebook TP1_regression.ipynb
```

Ou depuis Cursor : sélectionner le kernel **Python Environments...** puis
choisir l'environnement qui contient les dépendances installées.

---

## Imports entre TPs

Chaque notebook ajoute la racine du projet au `sys.path` :

```python
import sys, os
sys.path.append(os.path.abspath('..'))
from utils import kfold_cv, standardize, armijo
```

---

## Contenu de `utils.py`

| Section | Fonctions clés |
|---------|---------------|
| Covering Number | `greedy_epsilon_cover`, `covering_number_curve` |
| Line Search | `armijo`, `goldstein`, `wolfe`, `SelfAdaptiveLineSearch` |
| Validation | `kfold_split`, `kfold_cv`, `train_val_test_split` |
| Régularisation | `l1_regularization`, `l2_regularization`, `apply_l1_update`, `apply_l2_update` |
| Métriques | `accuracy`, `precision_recall_f1`, `classification_report` |
| Projection | `project_l2_ball` |
| Normalisation | `standardize`, `add_bias` |
| Plotting | `plot_losses`, `plot_multi_losses`, `plot_train_test` |
| Normes duales | `norm_l1`, `norm_l2`, `norm_linf`, `dual_norm` |
| Regret | `compute_regret`, `plot_regret` |

---

## Résumé des TPs

### TP1 — Régression polynomiale
Gradient numérique (3 schémas), line search (Armijo/Goldstein/Wolfe),
courbe biais-variance, cross-validation K-fold, régularisation Ridge.

### TP2 — Classification binaire par Perceptron
Sous-gradient, choix de direction, line search adaptée au cas non différentiable,
métriques de classification, biais-variance, régularisation L2.

### TP3 — Accélérateurs
Momentum, Nesterov, AdaGrad, RMSProp, Adam et Aitken appliqués
aux deux problèmes de TP1 et TP2. Même interface unifiée pour les deux cadres.

### TP4 — Cadres online et stochastique
Gradient en ligne projeté (régression), sous-gradient en ligne (classification),
SGD et SSGD avec mini-lots, estimation du regret, comparaison online vs batch.

### TP5 — Apprentissage en ligne supervisé
Perceptron standard et normalisé, Passive-Aggressive (PA/PA-I/PA-II),
Online Subgradient Descent, Hedge (experts), régularisation L1/L2 en ligne,
noyaux linéaire/polynomial/gaussien, normes duales.