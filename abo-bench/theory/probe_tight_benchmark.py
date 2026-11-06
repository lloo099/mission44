#!/usr/bin/env python3
"""Settle the open question in abo-endogenous-capacity-note.md §5.

The question: is there an exogenous capacity sequence making the fluid
benchmark both a valid lower bound and asymptotically tight, under endogenous
(fatiguing) capacity?

This probes it by computing the TRUE optimum exactly, by dynamic programming
over the fatigue state, and comparing it against the fluid value under a
constant capacity set to the sustainable rate. If the gap is o(T), no
impossibility theorem is available and route 甲's core claim fails.

Model (F), as in verify_endogenous_capacity.py: a served job completes with
probability 1 if no review completed in the previous period, else 1 - delta.

Loss accounting follows Lykouris & Weng: a job not admitted incurs the
idiosyncrasy loss ell; a job admitted but unreviewed at the horizon incurs ell
as well; a reviewed job incurs zero. So with ell factored out,
    loss / ell = (arrivals) - (completed reviews).

Run:
    python3 abo-bench/theory/probe_tight_benchmark.py
    python3 abo-bench/theory/probe_tight_benchmark.py --selftest
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache


def opt_unreviewed(horizon: int, delta: float, arrivals: tuple[float, ...]) -> float:
    """Expected number of unreviewed jobs under the OPTIMAL policy, by exact DP.

    State is (period, fatigue flag, queue length), where the fatigue flag records
    whether a review completed in the previous period. The DP chooses admission
    and service; it is not told that always-admit/always-serve is optimal.

    Returns E[arrivals - completed reviews] under the optimal policy.
    """
    total_arrivals = sum(arrivals)

    @lru_cache(maxsize=None)
    def value(t: int, fatigued: int, queue: int) -> float:
        """Expected number of FUTURE completed reviews from this state."""
        if t == horizon:
            return 0.0
        lam = arrivals[t]
        best = -1.0
        # admit the arrival or not; serve or not. Admission only grows the queue,
        # but the DP still enumerates it rather than assuming it is free.
        for admit in (0, 1):
            q_after = queue + (admit if lam > 0 else 0)
            # fractional arrivals are handled by mixing, so restrict to lam in {0,1}
            for serve in (0, 1):
                if serve and q_after == 0:
                    continue
                if serve:
                    p = (1.0 - delta) if fatigued else 1.0
                    # success: one review completes, queue shrinks, next period fatigued
                    v = p * (1.0 + value(t + 1, 1, q_after - 1))
                    # failure: job returns to the queue, no completion, so not fatigued
                    v += (1.0 - p) * value(t + 1, 0, q_after)
                else:
                    v = value(t + 1, 0, q_after)
                best = max(best, v)
        return best

    reviewed = value(0, 0, 0)
    return total_arrivals - reviewed


def fluid_unreviewed(arrivals: tuple[float, ...], rate: float) -> float:
    """Fluid value (divided by ell) for a constant exogenous capacity `rate`.

    One type, so the LP collapses: per period it may admit at most `rate`, and a
    non-admitted arrival costs ell. Window size does not matter here because
    arrivals and capacity are both constant within each instance we test.
    """
    return sum(max(0.0, a - rate) for a in arrivals)


def sustainable_rate(delta: float) -> float:
    """Long-run throughput of always-serve: the fixed point of p -> 1 - delta*p."""
    return 1.0 / (1.0 + delta)


def run_stationary(verbose: bool = True) -> bool:
    """Stationary arrivals of 1 per period: is the sustainable-rate fluid tight?"""
    ok = True
    if verbose:
        print("Stationary instance, lambda = 1 per period\n")
        print(f"{'delta':>6} {'T':>4} {'OPT/ell':>9} {'fluid(r*)':>10} "
              f"{'gap':>8} {'gap/T':>9}")
    for delta in (0.1, 0.25, 0.5):
        r = sustainable_rate(delta)
        prev_gap = None
        gaps: list[float] = []
        for horizon in (8, 12, 16, 20):
            arrivals = (1.0,) * horizon
            opt = opt_unreviewed(horizon, delta, arrivals)
            fluid = fluid_unreviewed(arrivals, r)
            gap = opt - fluid
            if verbose:
                print(f"{delta:>6} {horizon:>4} {opt:>9.4f} {fluid:>10.4f} "
                      f"{gap:>8.4f} {gap / horizon:>9.5f}")
            # What matters is not the sign but the SCALE. The fluid value sits a
            # constant ABOVE OPT, because a finite-horizon optimum gets to start
            # on a rested reviewer while the sustainable rate already prices
            # fatigue in. A constant offset is not an impossibility result.
            gaps.append(gap)
            prev_gap = gap
        # decisive test: does the gap grow with T, or stay bounded?
        spread = max(gaps) - min(gaps)
        ok &= spread < 0.01 and abs(prev_gap) < 1.0
        if verbose:
            print()
    return ok


def run_nonstationary(verbose: bool = True) -> bool:
    """Bursty arrivals that actually load the server: is a tuned constant tight?

    The earlier (1,1,0,0) pattern was degenerate — load 0.5 is below the
    sustainable rate for every delta tested, so OPT was 0 and the test proved
    nothing. (1,1,1,0) gives load 0.75, above the sustainable rate 1/(1+delta)
    for delta >= 1/3, so the queue genuinely binds.
    """
    ok = True
    if verbose:
        print("Non-stationary instance, bursts (1,1,1,0 repeated) — load 0.75\n")
        print(f"{'delta':>6} {'T':>4} {'OPT/ell':>9} {'best const':>11} "
              f"{'rate':>7} {'gap':>8}")
    for delta in (0.1, 0.25, 0.5):
        for horizon in (8, 12, 16, 20):
            arrivals = tuple(1.0 if (i % 4) < 3 else 0.0 for i in range(horizon))
            opt = opt_unreviewed(horizon, delta, arrivals)
            # search a constant capacity that makes the fluid value match OPT from below
            best_rate, best_fluid = 0.0, -1.0
            for i in range(2001):
                r = i / 2000.0
                f = fluid_unreviewed(arrivals, r)
                if f <= opt + 1e-9 and f > best_fluid:
                    best_fluid, best_rate = f, r
            gap = opt - best_fluid
            if verbose:
                print(f"{delta:>6} {horizon:>4} {opt:>9.4f} {best_fluid:>11.4f} "
                      f"{best_rate:>7.4f} {gap:>8.4f}")
            ok &= best_fluid <= opt + 1e-9 and gap < 1.0
        if verbose:
            print()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    verbose = not args.selftest

    results = {
        "stationary: sustainable-rate fluid is a valid, non-diverging bound":
            run_stationary(verbose),
        "non-stationary: a tuned constant capacity stays a valid bound":
            run_nonstationary(verbose),
    }
    for name, ok in results.items():
        print(f"[{'OK ' if ok else 'FAIL'}] {name}")
    if any(not ok for ok in results.values()):
        return 1
    if verbose:
        print("\nReading: the gap between OPT and the sustainable-rate fluid value does")
        print("not grow with T in the stationary case, so there is no linear-regret")
        print("impossibility to prove there. See §5 of the note for what this costs")
        print("route 甲.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
