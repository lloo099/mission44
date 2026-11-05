#!/usr/bin/env python3
"""Check 0, offline part: what must a trace carry before ABO can be measured?

Nothing here touches the network and nothing here needs a trace. It answers one
question so the online part of Check 0 can be scoped before any data is pulled:

    given a candidate data source, which attention units and which human-model
    parameters are computable, and which are not recoverable at all?

Run it:
    python3 abo-bench/trace/schema.py            # coverage report per source
    python3 abo-bench/trace/schema.py --selftest # invariants only, no output

The distinction this file exists to enforce
-------------------------------------------
Two different things get confused when people talk about "measuring attention":

  * An attention unit is produced by the SIMULATION. How many handoffs a policy
    sends a person is a property of that policy, not of history. The trace
    cannot contain it, because the trace only ever recorded one policy: the one
    the repository's humans actually ran.
  * A human-model parameter is estimated from the TRACE. Service rate, response
    latency and availability windows are properties of the person, and history
    is the only place to get them.

Mixing the two is how a benchmark ends up reporting its own assumptions back to
itself. Each entry below therefore declares which side it lives on, and an
attention unit additionally declares the trace fields needed to GROUND it
(the historical counterfactual it is scored against), which is not the same as
the fields needed to compute it.

Provenance of the design
------------------------
〔一手〕 marks behaviour read from deepseek-ai/dsh-superagent-prototype's own
docs (002-authorization, 003-execution, 007-architecture). 〔设计〕 marks our
own modelling choices, which are not claims about that system.
"""
from __future__ import annotations

import argparse
import sys

# --------------------------------------------------------------------------
# The normalized event record a trace must provide.
# --------------------------------------------------------------------------
# Field -> why ABO needs it. Keep this list minimal: every field added here is
# a field the online part of Check 0 must prove is obtainable.
TRACE_FIELDS: dict[str, str] = {
    "event_id":        "stable id; the Engine stores each Trigger once, so replay must dedupe 〔一手〕",
    "event_type":      "maps onto the candidate taxonomy in ../spec/event-mapping.md",
    "created_at":      "arrival time; drives queueing and every latency estimate",
    "repo":            "stream identity for the multi-repo difficulty tier",
    "thread_id":       "Evals are batched per Thread, so candidates must group by it 〔一手〕",
    "actor_login":     "who acted; needed to separate human action from bot and from agent",
    "actor_is_bot":    "bot messages are a distinct admission category 〔一手〕",
    "addressed_to":    "explicit request vs inferred candidate — the top split in the taxonomy 〔一手〕",
    "parent_event_id": "follow-up vs fresh request; also reconstructs thread order",
    "body_len":        "review burden proxy for a comment (NOT used by any policy feature — see note below)",
    "diff_added":      "review burden proxy for a PR: lines added",
    "diff_removed":    "review burden proxy for a PR: lines removed",
    "resolution":      "terminal state of the thread (merged / closed / stale) — the value signal",
    "resolved_at":     "when it terminated; with created_at gives time-to-resolution",
}

# body_len / diff_* quantify what a person must READ. They are deliberately
# absent from the policy's feature set: the index may only read host-computed
# trusted state, never anything derived from untrusted message text 〔一手
# invariant, see proposal §7.2〕. They are measurement inputs, not features.
NOT_POLICY_FEATURES = ("body_len", "diff_added", "diff_removed", "resolution", "resolved_at")

# --------------------------------------------------------------------------
# Attention units: produced by simulation, grounded against the trace.
# --------------------------------------------------------------------------
ATTENTION_UNITS: dict[str, dict] = {
    "items_requiring_action": {
        "what": "handoffs a policy puts in front of a person",
        "produced_by": "simulation",
        "grounding_fields": ("thread_id", "actor_login", "actor_is_bot", "created_at"),
        "note": "a Ticket hands off on a ready Report, a question, or blocked/failed 〔一手〕; "
                "grounded against whether a human in fact engaged that thread",
    },
    "notifications_delivered": {
        "what": "messages actually sent, including ones needing no action",
        "produced_by": "simulation",
        "grounding_fields": ("thread_id", "created_at"),
        "note": "distinct from the unit above: philosophy doc judges every message by its "
                "attention cost, not only the actionable ones 〔一手〕",
    },
    "review_volume": {
        "what": "how much a person must read, in diff lines and comment length",
        "produced_by": "simulation",
        "grounding_fields": ("diff_added", "diff_removed", "body_len"),
        "note": "the only unit whose magnitude comes straight from trace fields",
    },
    "context_switches": {
        "what": "distinct streams a person is pulled across per window",
        "produced_by": "simulation",
        "grounding_fields": ("thread_id", "repo", "actor_login", "created_at"),
        "note": "the quantity the practitioner accounts call the real cost; its WEIGHT is a "
                "separate problem — see switching_cost below",
    },
    "human_engaged_walltime": {
        "what": "wall-clock a person spends engaged",
        "produced_by": "simulation",
        "grounding_fields": ("actor_login", "created_at", "thread_id"),
        "note": "weakest unit: the trace shows when someone POSTED, not when they started "
                "reading. Treat as a lower bound, and say so in any result that uses it",
    },
}

# --------------------------------------------------------------------------
# Human-model parameters: estimated from the trace.
# --------------------------------------------------------------------------
HUMAN_MODEL_PARAMS: dict[str, dict] = {
    "service_rate": {
        "what": "items a person resolves per unit time",
        "required_fields": ("actor_login", "actor_is_bot", "created_at", "thread_id"),
        "recoverable": True,
    },
    "response_latency": {
        "what": "distribution of delay from arrival to human response",
        "required_fields": ("created_at", "parent_event_id", "actor_login", "actor_is_bot"),
        "recoverable": True,
    },
    "availability_windows": {
        "what": "when the person is actually reachable",
        "required_fields": ("actor_login", "created_at"),
        "recoverable": True,
    },
    "counterfactual_value": {
        "what": "did the work land, and did a human end up doing it themselves",
        "required_fields": ("resolution", "resolved_at", "actor_login", "thread_id"),
        "recoverable": True,
        "caveat": "PARTIAL. Gives ground truth for what history did, never for an action the "
                  "historical humans never took. This is the proposal's acknowledged soft spot",
    },
    "switching_cost": {
        "what": "cost of being pulled between streams, dominated by context reload time",
        "required_fields": (),
        "recoverable": False,
        "caveat": "NOT RECOVERABLE from any event trace. History records when someone posted, "
                  "never how long they spent reloading context first. Needs a literature prior "
                  "or a human study. A is a queueing model, so this is a hard dependency, not a "
                  "refinement",
    },
}

# --------------------------------------------------------------------------
# Candidate data sources. These are HYPOTHESES, to be settled by the online
# part of Check 0 — nothing below has been verified against a real pull.
# --------------------------------------------------------------------------
SOURCE_PROFILES: dict[str, dict] = {
    "gh_archive_bigquery": {
        "access": "needs a GCP project id; bq is installed but unconfigured in this environment",
        "verified": False,
        "assumed_fields": tuple(TRACE_FIELDS),
        "assumption": "event envelope plus payload is assumed to yield every field, with "
                      "thread_id and addressed_to derived rather than given",
    },
    "gh_archive_dump": {
        "access": "hourly json.gz over plain HTTP; data.gharchive.org not currently allowed",
        "verified": False,
        "assumed_fields": tuple(TRACE_FIELDS),
        "assumption": "same payloads as the BigQuery mirror, so the same derivations apply",
    },
    "github_rest_api": {
        "access": "reachable now, but this session's grant covers one repository only",
        "verified": False,
        "assumed_fields": tuple(f for f in TRACE_FIELDS if f != "event_id"),
        "assumption": "richer per-object detail than the event stream, but no single stable "
                      "event id across object types, so replay must synthesize one",
    },
}

DERIVED_NOT_GIVEN = ("thread_id", "addressed_to", "actor_is_bot", "resolution")


def coverage(profile_name: str) -> dict:
    """Which units and parameters a source can support, under its own assumptions."""
    profile = SOURCE_PROFILES[profile_name]
    have = set(profile["assumed_fields"])
    units = {
        name: sorted(set(spec["grounding_fields"]) - have)
        for name, spec in ATTENTION_UNITS.items()
    }
    params = {
        name: sorted(set(spec["required_fields"]) - have)
        for name, spec in HUMAN_MODEL_PARAMS.items()
        if spec["recoverable"]
    }
    return {"ungrounded_units": units, "unestimable_params": params}


def report() -> None:
    print("Check 0 (offline) — trace requirements for ABO\n")
    print(f"normalized fields required: {len(TRACE_FIELDS)}")
    print(f"attention units: {len(ATTENTION_UNITS)} (all produced by simulation, not read from trace)")
    print(f"human-model parameters: {len(HUMAN_MODEL_PARAMS)}\n")

    blocked = [n for n, s in HUMAN_MODEL_PARAMS.items() if not s["recoverable"]]
    print("NOT RECOVERABLE FROM ANY TRACE — needs a prior or a human study:")
    for n in blocked:
        print(f"  - {n}: {HUMAN_MODEL_PARAMS[n]['caveat']}")
    print()

    partial = [n for n, s in HUMAN_MODEL_PARAMS.items() if s.get("caveat") and s["recoverable"]]
    print("RECOVERABLE BUT PARTIAL:")
    for n in partial:
        print(f"  - {n}: {HUMAN_MODEL_PARAMS[n]['caveat']}")
    print()

    print(f"fields that must be DERIVED, not read off: {', '.join(DERIVED_NOT_GIVEN)}")
    print("  (each is a place the online part of Check 0 can fail quietly)\n")

    print("candidate sources — all UNVERIFIED, these are hypotheses:")
    for name, profile in SOURCE_PROFILES.items():
        cov = coverage(name)
        gaps = {k: v for k, v in {**cov["ungrounded_units"], **cov["unestimable_params"]}.items() if v}
        print(f"\n  {name}")
        print(f"    access: {profile['access']}")
        print(f"    assumption: {profile['assumption']}")
        print(f"    gaps under its own assumptions: {gaps or 'none'}")

    print("\nWhat the online part of Check 0 must then prove, in order:")
    print("  1. human response timing and action are recoverable for one repo, one week")
    print("  2. the four derived fields above can be derived without heuristics that beg the question")
    print("  3. switching_cost has a defensible prior, or A's queueing model is unidentified")


def selftest() -> None:
    for name, spec in ATTENTION_UNITS.items():
        unknown = set(spec["grounding_fields"]) - set(TRACE_FIELDS)
        assert not unknown, f"{name} grounds on undeclared fields: {unknown}"
        assert spec["produced_by"] == "simulation", f"{name} must be simulation-produced"
    for name, spec in HUMAN_MODEL_PARAMS.items():
        unknown = set(spec["required_fields"]) - set(TRACE_FIELDS)
        assert not unknown, f"{name} needs undeclared fields: {unknown}"
        if not spec["recoverable"]:
            assert not spec["required_fields"], f"{name} is unrecoverable, so it needs no fields"
            assert spec.get("caveat"), f"{name} is unrecoverable and must say why"
    assert not HUMAN_MODEL_PARAMS["switching_cost"]["recoverable"], \
        "switching_cost must stay marked unrecoverable; A's human model depends on it"
    assert HUMAN_MODEL_PARAMS["counterfactual_value"]["caveat"].startswith("PARTIAL")
    for field in NOT_POLICY_FEATURES:
        assert field in TRACE_FIELDS, f"{field} is named a non-feature but is not declared"
    for name in SOURCE_PROFILES:
        assert SOURCE_PROFILES[name]["verified"] is False, \
            f"{name} is marked verified, but no source has been checked against a real pull"
        assert set(SOURCE_PROFILES[name]["assumed_fields"]) <= set(TRACE_FIELDS)
    for field in DERIVED_NOT_GIVEN:
        assert field in TRACE_FIELDS
    print("[schema] selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="check invariants and exit")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        report()
    sys.exit(0)
