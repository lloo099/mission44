#!/usr/bin/env python3
"""Post-training evaluation: pass@1 on the GSM8K test set using the verifiable
reward as the judge. Uses vLLM for fast batched generation (GPU or NPU).

Usage:
    python3 eval/eval_gsm8k.py --model <ckpt-or-hf-id> --data data/gsm8k/test.parquet
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rewards"))
from gsm8k_verifier import compute_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint dir or HF model id")
    ap.add_argument("--data", default="data/gsm8k/test.parquet")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="0 = full test set")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.iloc[: args.limit]

    prompts = [row[0]["content"] for row in df["prompt"]]
    golds = [rm["ground_truth"] for rm in df["reward_model"]]

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model)  # vLLM picks cuda or npu (vllm-ascend) automatically
    # apply the model's chat template
    tok = llm.get_tokenizer()
    chats = [
        tok.apply_chat_template([{"role": "user", "content": p}],
                                add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
    outs = llm.generate(chats, sp)

    correct = 0
    for o, gold in zip(outs, golds):
        text = o.outputs[0].text
        correct += compute_score("gsm8k", text, gold)
    n = len(golds)
    print(f"[eval] model={args.model}")
    print(f"[eval] pass@1 = {correct}/{n} = {correct / n:.4f}")


if __name__ == "__main__":
    main()
