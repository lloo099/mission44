#!/usr/bin/env python3
"""M0 smoke test: detect the accelerator backend, count devices, and run one
tiny generation so you know the whole stack (torch + transformers + device) works
before you spend time on training.

It also captures a machine-readable provenance record. Hand-typing
`--hardware "1x Ascend 910B 64GB"` into the curve tool is how a claim drifts away
from the machine that produced it; `--json` reads the real device names and stack
versions off the host instead.

Usage:
    python3 env/check_env.py --model Qwen/Qwen2.5-0.5B-Instruct
    python3 env/check_env.py --skip-generate --json logs/run/env.json
"""
import argparse
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
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


def _run(cmd):
    """Best-effort shell capture; returns stripped stdout or None."""
    exe = cmd[0]
    if not shutil.which(exe):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
        return (out.stdout or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001
        return None


def cann_version():
    """CANN toolkit version from its install info file (or ASCEND_HOME_PATH)."""
    roots = [os.environ.get("ASCEND_TOOLKIT_HOME"), os.environ.get("ASCEND_HOME_PATH"),
             "/usr/local/Ascend/ascend-toolkit/latest"]
    for root in roots:
        if not root:
            continue
        for fname in ("ascend_toolkit_install.info", "version.cfg", "version.info"):
            path = os.path.join(root, fname)
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            m = re.search(r"version\s*[=:]\s*([^\s\n]+)", text, re.I)
            if m:
                return m.group(1)
    return None


def device_names(backend, count):
    """Real device model strings, so 'hardware' is observed rather than typed."""
    if backend == "npu":
        info = _run(["npu-smi", "info"])
        if info:
            names = re.findall(r"\b(910[A-Za-z0-9]*|310[A-Za-z0-9]*)\b", info)
            if names:
                return names[:count] if count else names
        try:
            import torch
            return [torch.npu.get_device_name(i) for i in range(count)]
        except Exception:  # noqa: BLE001
            return []
    if backend == "cuda":
        try:
            import torch
            return [torch.cuda.get_device_name(i) for i in range(count)]
        except Exception:  # noqa: BLE001
            return []
    return []


def capture(backend, count):
    """Provenance record: what hardware, what stack, which commit."""
    names = device_names(backend, count)
    uniq = sorted(set(names))
    hardware = f"{count}x {uniq[0]}" if (count and len(uniq) == 1) else (
        ", ".join(f"{names.count(n)}x {n}" for n in uniq) or None)

    stack = {
        "python": platform.python_version(),
        "torch": _pkg_version("torch"),
        "torch_npu": _pkg_version("torch_npu"),
        "cann": cann_version(),
        "transformers": _pkg_version("transformers"),
        "vllm": _pkg_version("vllm"),
        "vllm_ascend": _pkg_version("vllm-ascend"),
        "verl": _pkg_version("verl"),
        "ray": _pkg_version("ray"),
    }
    stack = {k: v for k, v in stack.items() if v}

    if backend == "npu":
        framework = "verl + vLLM-Ascend" if stack.get("vllm_ascend") else "verl"
    elif backend == "cuda":
        framework = "verl + vLLM" if stack.get("vllm") else "verl"
    else:
        framework = None

    here = os.path.dirname(os.path.abspath(__file__))
    sha = _run(["git", "-C", here, "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", here, "status", "--porcelain"])

    return {
        "capturedAt": dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
                        .isoformat().replace("+00:00", "Z"),
        "backend": backend,
        "deviceCount": count,
        "deviceNames": names,
        "hardware": hardware,
        "framework": framework,
        "stack": stack,
        "git": {"commit": sha, "dirty": bool(dirty)} if sha else None,
        "host": {"hostname": platform.node(), "platform": platform.platform()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="json_out", metavar="PATH",
                    help="write the provenance record here (feed it to logs_to_dashboard.py --env)")
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

    if args.json_out:
        rec = capture(backend, count)
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"[check] hardware:   {rec['hardware'] or '(unknown)'}")
        print(f"[check] framework:  {rec['framework'] or '(unknown)'}")
        print(f"[check] stack:      {', '.join(f'{k} {v}' for k, v in rec['stack'].items()) or '(none)'}")
        print(f"[check] provenance -> {args.json_out}")

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
