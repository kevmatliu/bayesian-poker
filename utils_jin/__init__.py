"""Utilities for the Bayesian poker implementation.

This package is intentionally separate from ``utils/`` so the original helper
code remains available.  The modules here turn parsed hand histories into the
small, discrete objects needed by filtering and EM:

* card/combo helpers for 1326 combos and 169 pre-flop classes;
* compact public state encoders;
* a simple GTO-inspired pre-flop action prior;
* a pre-flop Bayesian range filter;
* EM starter functions that turn filtered posteriors into soft counts.
"""

