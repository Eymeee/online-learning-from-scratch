# Implementation Order for Theory Additions

Repository: `Eymeee/online-learning-from-scratch`

## Goal
Add these concepts across the 5 TPs with minimal rework:
- Covering Number
- Self-adaptive choice of \(\epsilon\)
- VC-dimension

## Recommended Order

### 1. `utils.py`
Define the reusable abstractions first.

Why first:
- It is the shared layer used by all TPs.
- The README already positions `utils.py` as the common utilities module.
- The README already lists covering-number utilities and self-adaptive line search there, so this is the natural extension point.

Target outputs:
- `vc_dimension_*` helpers
- `covering_number_*` helpers
- `adaptive_epsilon_*` helpers
- shared plotting / reporting helpers

### 2. TP2
Lock down VC-dimension first in the simplest setting: binary linear classification.

Why second:
- VC-dimension is most natural for perceptron / linear separators.
- Easier to validate theoretically and empirically than in kernel settings.

Target outputs:
- theory section on VC-dimension of linear classifiers
- tiny empirical shattering demo
- optional empirical epsilon-cover on predictions or margins

### 3. TP1
Make covering numbers concrete in polynomial regression.

Why third:
- Covering numbers fit regression/function-class complexity well.
- Easy to visualize against polynomial degree and regularization.

Target outputs:
- empirical covering number on validation outputs
- covering-number curve vs epsilon
- compare fixed epsilon vs adaptive epsilon

### 4. TP4
Operationalize adaptive epsilon in online/stochastic learning.

Why fourth:
- Adaptive epsilon is most useful when the algorithm evolves over time.
- Good place to compare fixed versus adaptive schedules.

Target outputs:
- epsilon schedule depending on iteration / data scale
- reporting and comparison under adaptive precision
- optional regret-resolution plots

### 5. TP5
Integrate all three concepts in the advanced online supervised setting.

Why fifth:
- Combines linear online methods, kernels, and Hedge.
- Best synthesis point after the simpler TPs are already stable.

Target outputs:
- VC note for linear case
- careful treatment of kernels
- empirical covering numbers of predictions
- adaptive epsilon for comparisons and aggregation experiments

### 6. TP3
Retrofit adaptive epsilon into optimizer comparisons last.

Why last:
- TP3 is optimization-centric, not the cleanest entry point for theory.
- Easier to adapt once the shared utilities and experiment conventions are already defined.

Target outputs:
- adaptive stopping / resolution criteria
- compare accelerators at matched adaptive precision

## Step-by-step starting point

Start with **`utils.py`**, not a TP notebook.

### First bounded task for Codex
Ask Codex to do only this:
1. Inspect `utils.py`
2. Identify what already exists for:
   - covering number
   - self-adaptive epsilon / line search
   - any capacity / complexity helpers
3. Propose the smallest API extension needed for:
   - `vc_dimension_linear`
   - `empirical_covering_number`
   - `adaptive_epsilon_schedule`
4. Do **not** modify TP notebooks yet
5. Return only:
   - proposed function signatures
   - exact insertion points in `utils.py`
   - any naming conflicts or existing duplicates

## Minimal Codex prompt

```text
Open utils.py and audit the existing shared utilities.
I need the smallest clean extension plan for three theory concepts:
1) VC-dimension
2) covering number
3) self-adaptive epsilon

Tasks:
- Find what already exists related to covering number and adaptive epsilon / self-adaptive line search.
- Do not edit anything yet.
- Propose a minimal API to add in utils.py for:
  - vc_dimension_linear(d, bias=True)
  - empirical_covering_number(objects, epsilon, metric="l2")
  - adaptive_epsilon_schedule(t=None, n=None, scale=None, mode="sqrt")
- Tell me the exact insertion points in utils.py.
- Flag duplicates, conflicts, or places where existing code should be reused instead of rewritten.

Return a concise report only. No implementation yet.
```
