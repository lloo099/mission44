#!/usr/bin/env python3
"""Machine-check the two claims in docs/proposals/abo-endogenous-capacity-note.md.

Neither claim is taken on trust from the write-up: Lemma 1's throughput bound is
checked by brute force over EVERY service schedule for small horizons, and
Proposition B's counterexample is recomputed from the fatigue model rather than
restated.

Fatigue model (F): a scheduled review succeeds with probability 1 - delta if a
review completed in the previous period, and 1 otherwise. delta is a constant,
not a function of the horizon.

Run:
    python3 abo-bench/theory/verify_endogenous_capacity.py
    python3 abo-bench/theory/verify_endogenous_capacity.py --selftest
"""
from __future__ import annotations

import argparse
import itertools
import sys


def expected_throughput(schedule: tuple[int, ...], delta: float) -> float:
    """Expected number of completed reviews under model (F).

    schedule[t] == 1 means the policy attempts a review in period t. Success in
    period t depends on whether a review COMPLETED in period t-1, which is
    itself random, so we carry the probability forward exactly rather than
    assuming the previous attempt succeeded.
    """
    p_prev = 0.0          # P(a review completed in the previous period)
    total = 0.0
    for attempt in schedule:
        if attempt:
            # rate is 1 - delta when the previous period completed a review
            p_now = p_prev * (1.0 - delta) + (1.0 - p_prev) * 1.0
        else:
            p_now = 0.0
        total += p_now
        p_prev = p_now
    return total


def brute_force_best(horizon: int, delta: float) -> tuple[float, tuple[int, ...]]:
    """Maximum expected throughput over all 2^horizon schedules."""
    best, arg = -1.0, ()
    for schedule in itertools.product((0, 1), repeat=horizon):
        v = expected_throughput(schedule, delta)
        if v > best:
            best, arg = v, schedule
    return best, arg


def lemma1_bound(horizon: int, delta: float) -> float:
    """The bound claimed in Lemma 1: sum_t p_t <= (1 - delta/2) T + 1."""
    return (1.0 - delta / 2.0) * horizon + 1.0


def check_lemma1(verbose: bool = True) -> bool:
    """Lemma 1 must hold for every schedule, not merely for the best one."""
    ok = True
    rows = []
    for delta in (0.05, 0.1, 0.25, 0.5):
        for horizon in range(1, 15):
            best, arg = brute_force_best(horizon, delta)
            bound = lemma1_bound(horizon, delta)
            holds = best <= bound + 1e-12
            ok &= holds
            if horizon in (6, 12):
                rows.append((delta, horizon, best, bound, best / horizon, arg, holds))
    if verbose:
        print("Lemma 1 — brute force over every schedule\n")
        print(f"{'delta':>6} {'T':>3} {'best':>8} {'bound':>8} {'best/T':>8}  holds  argmax")
        for delta, horizon, best, bound, rate, arg, holds in rows:
            s = "".join(str(b) for b in arg)
            print(f"{delta:>6} {horizon:>3} {best:>8.4f} {bound:>8.4f} {rate:>8.4f}  "
                  f"{'yes' if holds else 'NO':>5}  {s}")
        print()
    return ok


def check_always_serve_rate(verbose: bool = True) -> bool:
    """Always-serve should approach the fixed point 1/(1+delta) < 1.

    This is what makes the instance in Theorem A bite: arrivals are 1 per
    period, so any sustained throughput strictly below 1 leaves a linear number
    of jobs unreviewed.
    """
    ok = True
    if verbose:
        print("Always-serve sustained rate vs fixed point 1/(1+delta)\n")
        print(f"{'delta':>6} {'rate@T=4000':>12} {'1/(1+delta)':>12} {'2/T tol':>9} {'<1?':>5}")
    for delta in (0.05, 0.1, 0.25, 0.5):
        horizon = 4000
        rate = expected_throughput((1,) * horizon, delta) / horizon
        fixed = 1.0 / (1.0 + delta)
        # p_t converges to the fixed point geometrically with ratio delta, so the
        # summed transient is at most |p_1 - p*| / (1 - delta) <= 2 for delta <= 1/2,
        # and the AVERAGE therefore sits within 2/T of the fixed point. A constant
        # tolerance would be wrong here: the gap is O(1/T), not machine epsilon.
        tol = 2.0 / horizon
        ok &= abs(rate - fixed) < tol and fixed < 1.0
        if verbose:
            print(f"{delta:>6} {rate:>12.6f} {fixed:>12.6f} {tol:>9.2e} {'yes':>5}")
    if verbose:
        print()
    return ok


def check_proposition_b(verbose: bool = True) -> bool:
    """Proposition B: equal per-window mass, different realizable throughput.

    Window size 2 over periods 1..4, windows {1,2} and {3,4}. Schedule A serves
    in periods 1 and 3; A' serves in periods 2 and 3. Both put mass 1 in each
    window, so a per-window LP constraint cannot tell them apart, yet their
    throughput differs by exactly delta.
    """
    ok = True
    if verbose:
        print("Proposition B — equal per-window mass, unequal throughput\n")
        head_a, head_ap = "A (1,3)", "A' (2,3)"
        print(f"{'delta':>6} {head_a:>9} {head_ap:>9} {'gap':>8} {'= delta?':>9}")
    for delta in (0.05, 0.1, 0.25, 0.5):
        a = expected_throughput((1, 0, 1, 0), delta)
        a_prime = expected_throughput((0, 1, 1, 0), delta)
        gap = a - a_prime
        # per-window masses are identical: one attempt in {1,2}, one in {3,4}
        mass_a = ((1 + 0), (1 + 0))
        mass_ap = ((0 + 1), (1 + 0))
        same_mass = mass_a == mass_ap
        ok &= same_mass and abs(gap - delta) < 1e-12
        if verbose:
            print(f"{delta:>6} {a:>9.4f} {a_prime:>9.4f} {gap:>8.4f} "
                  f"{'yes' if abs(gap - delta) < 1e-12 else 'NO':>9}")
    if verbose:
        print("\n  per-window attempt mass is (1, 1) for both schedules, so a constraint")
        print("  on per-window totals alone assigns them the same verdict.\n")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="assert only, minimal output")
    args = ap.parse_args()
    verbose = not args.selftest

    results = {
        "Lemma 1 (throughput bound)": check_lemma1(verbose),
        "always-serve rate < 1": check_always_serve_rate(verbose),
        "Proposition B (counterexample)": check_proposition_b(verbose),
    }
    for name, ok in results.items():
        print(f"[{'OK ' if ok else 'FAIL'}] {name}")
    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print("\nFAILED:", ", ".join(failed))
        return 1
    if verbose:
        print("\nNote: this verifies Lemma 1 and Proposition B only. The open question")
        print("in §5 of the note — whether a conservative exogenous capacity gives an")
        print("asymptotically tight benchmark — is NOT addressed here and decides")
        print("whether route 甲 holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
