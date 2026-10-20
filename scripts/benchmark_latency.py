"""Per-stage latency benchmark for the /analyze pipeline.

InferenceGuard sits between the user and the send button, so whatever it
costs is added to every message. This measures that cost per stage, on the
hardware it runs on, using the stage functions web/api.py itself calls
(presidio_stage, risk_stage, select_rewrite, rescore_stage, and the API's
utility evaluator). Nothing here is a copy of the pipeline.

Every message is pushed through every stage, including rewrite and rescore.
In the product a LOW-risk message skips those two, so "total" here is the
cost of a message that does get rewritten -- the slow path.

Usage:
    python scripts/benchmark_latency.py --rewriter heuristic --device cpu --runs 5
    python scripts/benchmark_latency.py --rewriter qwen3-1.7b --device cuda --load-in-4bit --runs 5

Writes artifacts/latency/<hardware>_<date>_<rewriter>.json and prints p50/p95
per stage. Text is never written to the output, only message index and length.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGES = ("presidio", "risk_model", "rewrite", "rescore", "utility", "total")

QWEN3_0_6B = "Qwen/Qwen3-0.6B"

# Synthetic, written for this benchmark. No real person's data. Lengths run
# from 10 to 120 words so the per-stage cost can be read against length.
MESSAGES: List[str] = [
    "Can you suggest a good book to read on a rainy weekend?",
    "I am 34 and just started running, how often should I train?",
    "My co-op ends in December and I take the Green Line to campus.",
    "As a nurse on night shifts, what should I eat to stay alert at work?",
    "I moved to Denver last spring and want to find a hiking group for beginners near the foothills.",
    "Help me write a polite email to my landlord about a leaking kitchen tap that has not been fixed for two weeks.",
    "I finished my master's in chemistry last year and now teach high school science. How do I keep "
    "students engaged in the last period of the day?",
    "We just had our second child and my partner returns to work next month. What are some realistic "
    "ways to split night feeds so neither of us burns out completely?",
    "I work as a software engineer at a startup and my manager wants a design doc for the new billing "
    "service by Friday. Can you outline the sections I should include and what each one covers?",
    "Planning a trip from Seattle to Vancouver by train in late October. What should I pack, and is it "
    "worth getting a rail pass if I also want to visit Victoria for a day or two?",
    "My grandmother is turning ninety and the whole family is flying in. I need ideas for a small party "
    "at her retirement home that does not tire her out, keeps the grandchildren busy, and fits into "
    "about two hours in the afternoon.",
    "I am a second-year law student and my study group keeps falling apart before exams. Everyone has "
    "part-time jobs and different class schedules. How can we organise our time so that we actually "
    "cover contracts and torts properly before finals in May?",
    "After fifteen years as an accountant I want to retrain as a data analyst. I know Excel very well "
    "and a bit of SQL from work. Which skills should I learn first, how long does the switch usually "
    "take, and should I do a bootcamp or teach myself online in the evenings?",
    "Our school district is cutting the music program and I coach the middle school band. I want to "
    "write a short speech for the board meeting next Tuesday that explains why the program matters, "
    "with two or three concrete examples from students, without sounding angry or defensive.",
    "I just got back from my morning shift at the hospital and I cannot sleep. I have been doing "
    "rotating shifts for three years now, two weeks of days and two weeks of nights. My doctor said "
    "to keep a regular schedule, but that is impossible with this rota. What else can I try?",
    "I am 27, living in Austin, and thinking about buying my first condo instead of renting. Prices "
    "have gone up a lot since I graduated from UT. I earn a decent salary in marketing but I also "
    "have student loans. How should I think about whether buying makes sense right now, and what "
    "should I ask a lender in the first meeting?",
    "Write a short bedtime story for a six-year-old about a lighthouse keeper's cat who is afraid of "
    "the dark but has to guide a lost boat home during a storm. Keep the language simple, make it "
    "about three hundred words, and end with the cat curling up by the lamp feeling brave and proud "
    "of what she did for the sailors that night.",
    "I retired from the post office two years ago and I have more free time than I expected. My wife "
    "still works part time at the library. I used to enjoy woodworking in my twenties, and I am "
    "thinking about setting up a small workshop in the garage. What tools should a beginner buy "
    "first, how much space do I really need, and what safety rules matter most for someone my age?",
    "My team at the consultancy is moving to a four-day week as a trial for the next quarter. I lead "
    "a group of six analysts across two time zones, and I am worried client deadlines will slip. "
    "Can you help me draft a plan for how we cover client calls, who is on call on the fifth day, "
    "how we track whether the trial is working, and what I should say to the partners if the "
    "numbers look worse in the first month than they do later on?",
    "I graduated from a small liberal arts college with a degree in history and have spent the last "
    "eight years working in museum education, mostly running school programs and summer camps for "
    "kids. Funding has been cut again and I am thinking about moving into instructional design for "
    "companies. I have never worked in a corporate setting. What does the day-to-day work look like, "
    "which of my skills transfer directly, what portfolio pieces would convince a hiring manager, "
    "and how do I explain the move in a cover letter without sounding like I am running away from "
    "a sector I still care about deeply?",
]


def _word_count(text: str) -> int:
    return len(text.split())


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    xs = sorted(values)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def hardware_info(device: str) -> Dict[str, Any]:
    import torch
    import transformers

    info: Dict[str, Any] = {
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "gpu": None,
    }
    if device == "cuda" and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info


def peak_rss_mb() -> Optional[float]:
    """Peak resident memory of this process so far (all loaded models included)."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # bytes on macOS, kilobytes on Linux
    return round(rss / 1e6 if sys.platform == "darwin" else rss / 1e3, 1)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_risk_model(api: Any, checkpoint: Optional[str], device: str, use_rules: bool) -> tuple:
    """Return (risk_fn or None, description)."""
    if use_rules:
        return None, "rule-based estimate_privacy_risk (no ModernBERT)"

    from transformers import AutoTokenizer

    from src.risk_model.model import ModernBertRiskClassifier

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    if checkpoint and Path(checkpoint).exists():
        model = ModernBertRiskClassifier.from_pretrained(checkpoint)
        desc = f"trained checkpoint {checkpoint}"
    else:
        # Same architecture and size as the trained classifier, untrained
        # heads. Latency is representative; the scores are not.
        model = ModernBertRiskClassifier(model_name="answerdotai/ModernBERT-base")
        desc = "answerdotai/ModernBERT-base backbone, untrained heads (latency only)"
    model = model.to(device)
    model.eval()
    return api.make_risk_estimator(model, tokenizer, device), desc


def load_rewriter(name: str, device: str, load_in_4bit: bool, adapter_path: Optional[str]) -> tuple:
    """Return (QwenRewriterInference or None, description dict)."""
    if name == "heuristic":
        return None, {"rewriter": "heuristic", "model_id": None, "adapter_loaded": False}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.rewriter.inference import DEFAULT_BASE_MODEL, QwenRewriterInference

    model_id = {"qwen3-0.6b": QWEN3_0_6B, "qwen3-1.7b": DEFAULT_BASE_MODEL}[name]
    rw = QwenRewriterInference(
        model_name_or_path=model_id,
        adapter_path=adapter_path,
        device=device,
        load_in_4bit=load_in_4bit,
    )
    if rw.is_adapter_loaded:
        return rw, {"rewriter": name, "model_id": rw.resolved_base_model, "adapter_loaded": True,
                    "load_in_4bit": rw.load_in_4bit}

    # No adapter in the repo yet. Load the base weights into the same class so
    # rewrite() runs the real prompt, generation settings and output cleaning.
    # LoRA adds a small per-token cost on top of this when unmerged.
    is_cuda = device == "cuda" and torch.cuda.is_available()
    quant = None
    if load_in_4bit and is_cuda:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=rw.torch_dtype,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=rw.torch_dtype,
        quantization_config=quant,
        device_map="auto" if is_cuda else None,
    )
    if not is_cuda:
        model = model.to(device)
    model.eval()
    rw.model, rw.tokenizer, rw._is_adapter_loaded = model, tokenizer, True
    return rw, {
        "rewriter": name,
        "model_id": model_id,
        "adapter_loaded": False,
        "load_in_4bit": bool(quant),
        "dtype": str(rw.torch_dtype),
        "rewriter_note": "base weights without a LoRA adapter: generation latency only, output quality not meaningful",
    }


def run_once(api: Any, text: str) -> Dict[str, Any]:
    """One message through every stage; returns ms per stage plus labels."""
    from src.product.thresholds import PROVISIONAL_MAX_RISK_THRESHOLD

    timings: Dict[str, float] = {}
    t_total = time.perf_counter()

    t = time.perf_counter()
    spans, presidio_text = api.presidio_stage(text)
    timings["presidio"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    _, before = api.risk_stage(text, f"bench-{uuid.uuid4().hex}", spans)
    timings["risk_model"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    rewritten, engine = api.select_rewrite(text, presidio_text, None, PROVISIONAL_MAX_RISK_THRESHOLD)
    timings["rewrite"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    api.rescore_stage(rewritten)
    timings["rescore"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    utility = api.utility_evaluator.compute_metrics(text, rewritten)
    timings["utility"] = (time.perf_counter() - t) * 1000

    timings["total"] = (time.perf_counter() - t_total) * 1000
    return {
        "timings": timings,
        "band": str(getattr(before.inferential_band, "value", before.inferential_band)),
        "engine": engine,
        "changed": api.rewrite_changed(text, rewritten),
        "utility_backend": utility.get("utility_backend"),
    }


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rewriter", choices=("heuristic", "qwen3-0.6b", "qwen3-1.7b"), default="heuristic")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--load-in-4bit", action="store_true", help="NF4 rewriter weights (CUDA only)")
    parser.add_argument("--runs", type=int, default=5, help="Measured passes over the message list")
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured passes before measuring")
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter dir; base weights are used if absent")
    parser.add_argument("--risk-checkpoint", default="artifacts/risk_model/best_model")
    parser.add_argument("--no-risk-model", action="store_true", help="Use the rule-based risk estimate")
    parser.add_argument("--out-dir", default="artifacts/latency")
    args = parser.parse_args(argv)

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")

    from web import api  # loads the same module-level services the server does

    risk_fn, risk_desc = load_risk_model(api, args.risk_checkpoint, args.device, args.no_risk_model)
    api.risk_model = risk_fn
    api.conversation_tracker = api.ConversationTracker(risk_estimator_fn=risk_fn)
    rewriter, rewriter_desc = load_rewriter(args.rewriter, args.device, args.load_in_4bit, args.adapter_path)
    api.qwen_rewriter = rewriter

    for _ in range(args.warmup):
        for text in MESSAGES:
            run_once(api, text)
    if args.device == "cuda":
        torch.cuda.synchronize()

    samples: Dict[str, List[float]] = {s: [] for s in STAGES}
    per_message: List[Dict[str, Any]] = []
    for i, text in enumerate(MESSAGES):
        per_message.append({"index": i, "words": _word_count(text), "total_ms": [], "band": None,
                            "engine": None, "changed": None})
    for _ in range(args.runs):
        for i, text in enumerate(MESSAGES):
            res = run_once(api, text)
            for stage in STAGES:
                samples[stage].append(res["timings"][stage])
            rec = per_message[i]
            rec["total_ms"].append(round(res["timings"]["total"], 3))
            rec["band"], rec["engine"], rec["changed"] = res["band"], res["engine"], res["changed"]
            utility_backend = res["utility_backend"]

    stage_stats = {
        stage: {
            "p50": round(percentile(vals, 50), 3),
            "p95": round(percentile(vals, 95), 3),
            "mean": round(sum(vals) / len(vals), 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "n": len(vals),
        }
        for stage, vals in samples.items()
    }

    hw = hardware_info(args.device)
    hw_name = hw["gpu"] if hw.get("gpu") else hw["cpu"]
    today = dt.date.today().isoformat()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_slug(args.device + '-' + hw_name)}_{today}_{args.rewriter}.json"

    report = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "command": "python scripts/benchmark_latency.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": hw,
        "config": {
            **rewriter_desc,
            "risk_model": risk_desc,
            "presidio_backend": "presidio" if api.presidio_baseline.analyzer is not None else "regex fallback",
            "utility_backend": utility_backend,
            "runs": args.runs,
            "warmup": args.warmup,
            "messages": len(MESSAGES),
            "message_words_min": min(_word_count(m) for m in MESSAGES),
            "message_words_max": max(_word_count(m) for m in MESSAGES),
            "benchmark_note": "every message runs every stage; LOW-risk messages skip rewrite and rescore in /analyze",
        },
        "peak_rss_mb": peak_rss_mb(),
        "stages_ms": stage_stats,
        "per_message": per_message,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{hw_name} | device={args.device} | rewriter={args.rewriter} | "
          f"risk={risk_desc} | runs={args.runs} x {len(MESSAGES)} messages")
    print(f"{'stage':<12}{'p50 ms':>12}{'p95 ms':>12}")
    for stage in STAGES:
        print(f"{stage:<12}{stage_stats[stage]['p50']:>12.2f}{stage_stats[stage]['p95']:>12.2f}")
    print(f"\nwrote {out_path}")
    return report


if __name__ == "__main__":
    main()
