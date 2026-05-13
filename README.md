# Bayesian Poker

Using the Pluribus dataset, we conducted Bayesian range filtering and determined latent player tendencies using EM and Newton. The model is comprised of the following:
- Population-level action baselines (`beta`), trained using multinomial logistic regression
- Per-player tendency tilts for each action (`theta`) from the global population-level baselines, learned via EM or Newton's method
- Range filter `R_t`, a 169-dimensional probability vector on the canonical hand equivalence classes in the preflop regime -> explodes into 1326-dimensional probability vector in postflop for each combo, updated via the predicted actions following `beta` and `theta`.

## `utils/`

| Module | Role |
|--------|------|
| `parse.py` | `.phh` → `Hand` (states, actions, hole cards). |
| `prior/` | Preflop/postflop feature keys, `*Prior` models, shared SGD training (`training.py`). |
| `em/` | Preflop/postflop E-step / M-step over hand classes or combos (`common.py` shared bits). |
| `newton/` | Marginal log-likelihood + L2 MAP Newton updates for `theta` (preflop/postflop + shared helpers). |
| `action/` | Action-phase tagging, preflop/postflop action models and context builders; softmax / utility vectors. |
| `strength/` | Preflop equivalence classes; determining postflop strength and `fast_eval` tables. |
| `filter/` | Range narrowing (preflop classes, postflop combos) against priors + thetas. |
| `postflop_runner_bridge.py` | From `Hand`/`State` to features, strength cache, EM bundles. |
| `tendency.py` | `TendencyTheta`, `ActionPrior`, `InferencePhase` — shared types wiring priors + tilts. |
| `eval/` | Helpers for notebooks: global priors / player thetas / ranges, Brier, tables, plots, online CSV. |

## `artifacts/` (JSON, CSV)

| File | Contents |
|------|----------|
| `global_priors.json` | Schema `bayesian_poker.global_priors.v1`: trained `beta`, dimensions, column labels, session metadata. |
| `player_thetas_em.json` | Schema `bayesian_poker.player_thetas.v1`: per-player `theta_pre` / `theta_post` from EM. |
| `player_thetas_newton.json` | Same schema; thetas from marginal Newton (`runners.find_theta --newton`). |
| `filter_sessions_range_history.csv` | 1331-column CSV that contains the range for each street and observer-target pair, given a session and hand |

`global_priors_eval_supervised.npz` is a cached supervised eval tensor, not JSON.

## `runners/`

| Script | Purpose |
|--------|---------|
| `train.py` | Fit `global_priors.json` from labeled rows in `.phh` corpora. |
| `find_theta.py` | Load global priors, build bundles, infer `theta` per player (`--em` or `--newton`). |
| `filter_sessions.py` | Batch preflop/postflop range filtering; consumes priors + player thetas JSON. |
| `execute.py` | Shared filter + EM helpers for `filter_sessions`. |
| `common.py` | Paths, session expansion, supervised rows, splits, JSON IO helpers. |
| `models.py` | Dataclasses / config shared by execute and common. |

CLI help: `python -m runners.<module> -h`.

## `eval_nbs/`

Jupyter notebooks that consume artifact logs and evaluating models using `utils/eval` helpers: `global_priors_evaluation.ipynb`, `player_thetas_evaluation.ipynb`, `range_evaluation.ipynb`.

## Reproducing experiments (`command_*.sh`)

Root-level wrappers from repo root. Typical order: train priors → fit thetas → run filtering.

| Script | What it runs |
|--------|----------------|
| `command_train.sh` | `runners.train` -> `artifacts/global_priors.json` (default: `pluribus/`, `sessions_train.txt`). |
| `command_theta.sh` | Alias for `command_theta_em.sh`. |
| `command_theta_em.sh` | `runners.find_theta --em` -> `artifacts/player_thetas_em.json` (default players / sessions in script header). |
| `command_theta_newton.sh` | `runners.find_theta --newton` -> `artifacts/player_thetas_newton.json`. |
| `command_filter.sh` | `runners.filter_sessions` with **Newton** thetas; default range CSV name under `artifacts/` (see script). |
| `command_filter_em.sh` | Same as `command_filter.sh` but **EM** thetas and EM-named range CSV. |

Pass `[PLURIBUS_ROOT]`, `[SESSIONS_FILE]`, optional players, then `--` for extra flags to the Python module (see each script’s header).

## Quick CLI (without shell wrappers)

```bash
python -m runners.train pluribus/ --out artifacts/global_priors.json
python -m runners.find_theta pluribus/ --global-priors artifacts/global_priors.json --out artifacts/player_thetas_em.json
python -m runners.filter_sessions -h
```

## Tests

```bash
python -m pytest tests/
```

## Data layout

**Session folder**: numbered `.phh` files (one hand per file). **Corpus root**: directory whose *children* are session folders. `runners.common.expand_data_path` treats a single session vs a root accordingly.
