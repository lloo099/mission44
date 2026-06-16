#!/usr/bin/env python3
"""M0 smoke test: detect the accelerator backend, count devices, and run one
tiny generation so you know the whole stack (torch + transformers + device) works
before you spend time on training.

Usage:
    python3 env/check_env.py --model Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import sys


def detect_backend():
    """Return ('npu'|'cuda'|'cpu', device_count)."""
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"[check] torch import failed: {e}", file=sys.stderr)
        return "none", 0

    # Ascend NPU (torch_npu registers the 'npu' backend)
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return "npu", torch.npu.device_count()
    except Exception:  # noqa: BLE001
        pass

    if torch.cuda.is_available():
        return "cuda", torch.cuda.device_count()
    return "cpu", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="What is 17 + 26? Answer with the number only.")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--skip-generate", action="store_true",
                    help="only report devices, don't load the model")
    args = ap.parse_args()

    backend, count = detect_backend()
    print(f"device backend: {backend}")
    print(f"device count:   {count}")
    if backend in ("cpu", "none"):
        print("[check] WARN: no accelerator detected — training will be unusably slow.")

    if args.skip_generate:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = {"npu": "npu", "cuda": "cuda"}.get(backend, "cpu")
    print(f"[check] loading {args.model} on {device} …")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32
    ).to(device)

    msgs = [{"role": "user", "content": args.prompt}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)
    out = model.generate(inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    text = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"[check] prompt:     {args.prompt}")
    print(f"[check] completion: {text.strip()!r}")
    print("[check] OK — stack works end to end.")


if __name__ == "__main__":
    main()
