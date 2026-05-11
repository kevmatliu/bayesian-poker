# Bayesian Poker

Population-level action baselines and per-player **tendency** parameters inferred from Pluribus-style `.phh` hand histories.

## What the model is

1. **Global priors (`beta`)** — Multinomial logistic regression over engineered features:
   - **Preflop**: 169 abstract hand classes × table state (position, SPR bucket, facing bet, etc.).
   - **Postflop**: Two heads — **facing a bet** (fold / call / raise) and **no facing bet** (call vs raise only). Features combine made/draw strength, pot geometry, position, board texture, rich per-combo categoricals (Method A), and optional Monte Carlo rollout equity on the flop (Method E).

2. **Player or session tilt (`theta`)** — A 3-vector that tilts log-probabilities toward actions whose **behavior vectors** deviate from the population baseline expectation. Same mathematical idea preflop (`theta_pre`) and postflop (`theta_post`).

3. **EM when hole cards are hidden** — For an observer watching a target, the target’s private cards are unknown. The code alternates:
   - **E-step**: posterior over hand classes (preflop, 169 keys) or concrete combos (postflop, up to 1,326 keys) using the current `theta` and frozen `beta`.
   - **M-step**: gradient ascent on `theta` with an L2 penalty, using expected sufficient statistics from the E-step.

Supervised training of `beta` uses only rows where the actor’s hole cards are known (e.g. labeled Pluribus exports).

## Repository layout

| Path | Role |
|------|------|
| `train.py` | Fit `artifacts/global_priors.json` from `.phh` corpora. |
| `find_theta.py` | Load global priors, build EM bundles, run preflop + postflop EM per player. |
| `pipeline_common.py` | Hand loading, session expansion, supervised row collection, EM bundle gathering, splits. |
| `utils/parse.py` | `.phh` → `Hand` object (states, actions, hole cards). |
| `utils/prior/preflop.py` | Preflop `StateKey`, `preflop_feature_vector`, `PreflopPrior`, baseline training. |
| `utils/prior/postflop.py` | `PostflopFeatures`, `PostflopPrior`, vectorized `action_probs_matrix` (Method D). |
| `utils/prior/training.py` | Generic multinomial logit SGD for 2- and 3-class baselines. |
| `utils/em/preflop.py` | Preflop E/M over 169 classes. |
| `utils/em/postflop.py` | Postflop E/M over combo keys; batched likelihoods. |
| `utils/postflop_runner_bridge.py` | From `Hand`/`State` to features, strength cache, EM bundles. |
| `utils/strength/` | Preflop equivalence classes; postflop strength, `fast_eval` tables. |
| `utils/filter/` | Range narrowing (preflop classes, postflop combos) for priors. |

## Data layout

- **Session folder**: contains numbered `.phh` files (one hand per file).
- **Pluribus root**: directory whose *children* are session folders (each child holds `.phh` files). `pipeline_common.expand_data_path` distinguishes a single session from a root.

## Commands

```bash
# Fit global baselines (example paths)
python train.py pluribus/ --out artifacts/global_priors.json

# Infer per-player theta (requires global_priors.json)
python find_theta.py pluribus/ --global-priors artifacts/global_priors.json --out artifacts/player_thetas.json
```

Use `python train.py -h` and `python find_theta.py -h` for session filters, learning rates, EM iterations, and postflop equity MC sample counts.

## Tests

```bash
python -m pytest tests/
```

## Artifact schema

`global_priors.json` uses schema key `bayesian_poker.global_priors.v1` and stores `preflop.beta_preflop`, `postflop.beta_facing`, `postflop.beta_no_bet`, feature dimensions, and column labels for traceability.

Postflop `beta` matrices trained at older feature dimensions are **right-padded with zeros** when loaded so extra feature columns behave as unused until retrained (`utils/prior/postflop._coerce_beta_to_phi_dim`).
