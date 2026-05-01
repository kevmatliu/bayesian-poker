# utils_jin

This folder contains the first implementation layer for the Bayesian poker
model:

1. `cards.py`: 1326 combo and 169 hand-class helpers.
2. `action_encoder.py`: pot-before-action action bucketing.
3. `state_encoder.py`: compact public state keys.
4. `records.py`: flat decision-record extraction from the existing parser.
5. `gto_prior.py`: simple GTO-inspired pre-flop `p(a | h, s)`.
6. `preflop_filter.py`: Bayesian pre-flop range filtering.
7. `em_preflop.py`: E-step, soft counts, and M-step helpers for starting EM.

Minimal smoke-test usage:

```python
from utils_jin.records import extract_session_records, group_records_by_player_hand
from utils_jin.preflop_filter import filter_preflop_records

records = extract_session_records("pluribus/78", include_postflop=False)
groups = group_records_by_player_hand(records)

key, one_player_hand = next(iter(groups.items()))
filt = filter_preflop_records(one_player_hand)

print(key)
print(filt.top_k(10))
print(filt.steps[-1])
```

EM starter usage:

```python
from utils_jin.em_preflop import e_step_preflop, expected_action_counts, m_step_likelihoods

q = e_step_preflop(records)
counts = expected_action_counts(records, q)
phi = m_step_likelihoods(counts)
```

The GTO prior here is intentionally heuristic.  It is a clean interface for
plugging in real solver frequencies later.

Technical note: pre-flop bet/raise buckets use big-blind multiples, while
post-flop bet/raise buckets use pot fractions.  This is intentional: measuring
ordinary pre-flop opens against the blind pot makes almost every open look like
a large bet.
