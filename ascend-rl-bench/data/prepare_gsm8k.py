#!/usr/bin/env python3
"""Download GSM8K and write it in the parquet schema verl expects.

Each row:
    data_source:  "gsm8k"
    prompt:       [{"role": "user", "content": <question + instruction>}]
    ability:      "math"
    reward_model: {"style": "rule", "ground_truth": <gold answer string>}
    extra_info:   {"split": ..., "index": ..., "answer": <full solution>}

Usage:
    python3 data/prepare_gsm8k.py --out data/gsm8k
"""
import argparse
import os

INSTRUCTION = (
    "Solve the problem step by step. "
    "Put the final numeric answer on its own line after '#### '."
)


def to_row(example, split, idx):
    question = example["question"]
    answer = example["answer"]  # full solution; gold number is after '####'
    gold = answer.split("####")[-1].strip().replace(",", "")
    return {
        "data_source": "gsm8k",
        "prompt": [{"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": gold},
        "extra_info": {"split": split, "index": idx, "answer": answer},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gsm8k", help="output dir for {train,test}.parquet")
    ap.add_argument("--config", default="main", help="GSM8K config (main|socratic)")
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset("openai/gsm8k", args.config)

    for split in ("train", "test"):
        rows = [to_row(ex, split, i) for i, ex in enumerate(ds[split])]
        from datasets import Dataset
        out = os.path.join(args.out, f"{split}.parquet")
        Dataset.from_list(rows).to_parquet(out)
        print(f"[data] wrote {len(rows):>5} rows -> {out}")

    print("[data] done.")


if __name__ == "__main__":
    main()
