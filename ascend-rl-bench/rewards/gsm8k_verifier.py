#!/usr/bin/env python3
"""Verifiable reward for GSM8K (RLVR-style): extract the model's final answer and
exact-match it against the gold answer. No reward model, no human labels.

verl picks this up via:
    custom_reward_function.path=rewards/gsm8k_verifier.py
    custom_reward_function.name=compute_score

The verl reward signature is:
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

Run `python3 rewards/gsm8k_verifier.py --selftest` to verify the extractor logic
without any GPU / framework.
"""
import argparse
import re
import sys

# matches "#### 42", "#### -3.5", "#### 1,024"
_HASH = re.compile(r"####\s*(-?[0-9][0-9,]*\.?[0-9]*)")
# any number anywhere (used as a fallback / for free-form model output)
_NUM = re.compile(r"-?[0-9][0-9,]*\.?[0-9]*")


def _normalize(num: str) -> str:
    """Canonicalize a numeric string: strip commas, trailing zeros, '+'."""
    num = num.replace(",", "").replace("+", "").strip()
    try:
        f = float(num)
        # integers render without a trailing ".0"
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return num


def extract_gold(ground_truth: str) -> str:
    """GSM8K gold answers carry the final number after '####'."""
    m = _HASH.search(ground_truth)
    raw = m.group(1) if m else ground_truth
    return _normalize(raw)


def extract_pred(solution_str: str, method: str = "flexible") -> str | None:
    """Pull the model's final numeric answer.
    - strict:   only accept an explicit '#### <num>'
    - flexible: prefer '#### <num>', else take the LAST number in the text
    """
    m = _HASH.search(solution_str)
    if m:
        return _normalize(m.group(1))
    if method == "strict":
        return None
    nums = _NUM.findall(solution_str)
    return _normalize(nums[-1]) if nums else None


def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  method: str = "flexible", correct: float = 1.0,
                  format_bonus: float = 0.0) -> float:
    """Return reward for one rollout. 1.0 if the final answer matches, else 0.0.
    Optionally add a small `format_bonus` when the model used the '#### <num>' form.
    """
    pred = extract_pred(solution_str or "", method=method)
    gold = extract_gold(ground_truth or "")
    if pred is None:
        return 0.0
    score = correct if pred == gold else 0.0
    if format_bonus and _HASH.search(solution_str or ""):
        score += format_bonus
    return score


# ---------------------------------------------------------------- self-test
_CASES = [
    # (solution, ground_truth, expected)
    ("The total is #### 42", "Jenny... #### 42", 1.0),
    ("so the answer is 42.", "#### 42", 1.0),
    ("I think it's 7", "#### 42", 0.0),
    ("answer: 1,024", "#### 1024", 1.0),
    ("result #### -3", "#### -3", 1.0),
    ("no numbers here", "#### 5", 0.0),
    ("first 10 then finally 18 apples", "#### 18", 1.0),  # last-number fallback
]


def _selftest() -> int:
    bad = 0
    for sol, gt, exp in _CASES:
        got = compute_score("gsm8k", sol, gt)
        ok = got == exp
        bad += not ok
        print(f"[{'ok' if ok else 'FAIL'}] exp={exp} got={got}  sol={sol!r}")
    print(f"\n{'ALL PASS' if not bad else f'{bad} FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    ap.print_help()
