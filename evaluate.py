"""Run the experiment: accuracy WITH Wikipedia evidence vs. WITHOUT evidence.

Usage:
    python evaluate.py              # all claims, both conditions
    python evaluate.py --limit 5    # quick, cheap trial run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import anthropic

from factcheck import LABELS, MODEL, check_claim, fetch_wikipedia

ROOT = Path(__file__).parent
CONDITIONS = {"with_evidence": True, "no_evidence": False}


def run_one(item: dict, use_evidence: bool, effort: str) -> dict:
    try:
        r = check_claim(item["claim"], page=item["page"], use_evidence=use_evidence, effort=effort)
        usage = r.get("usage")
        tokens = (usage.input_tokens, usage.output_tokens) if usage else (0, 0)
        pred, reason = r["label"], r["reason"]
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
        raise  # a bad key affects every claim; stop instead of scoring 0%
    except Exception as e:  # keep the run going; count as wrong
        pred, reason, tokens = "ERROR", f"{type(e).__name__}: {e}", (0, 0)
    return {
        "id": item["id"], "claim": item["claim"], "gold": item["label"],
        "pred": pred, "correct": pred == item["label"], "reason": reason,
        "input_tokens": tokens[0], "output_tokens": tokens[1],
    }


def report(name: str, rows: list[dict]) -> str:
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    lines = [f"== {name}: accuracy {correct}/{n} = {correct / n:.0%}"]
    for label in LABELS:
        sub = [r for r in rows if r["gold"] == label]
        if sub:
            c = sum(r["correct"] for r in sub)
            lines.append(f"   {label:<16} {c}/{len(sub)} = {c / len(sub):.0%}")

    # Confusion matrix: rows = gold label, columns = predicted label
    cols = LABELS + (["ERROR"] if any(r["pred"] == "ERROR" for r in rows) else [])
    short = {"Supported": "SUP", "Refuted": "REF", "Not Enough Info": "NEI", "ERROR": "ERR"}
    lines.append("   confusion (rows=gold, cols=pred): " + " ".join(f"{short[c]:>4}" for c in cols))
    for g in LABELS:
        counts = [sum(1 for r in rows if r["gold"] == g and r["pred"] == c) for c in cols]
        lines.append(f"   {'':<33}{short[g]}: " + " ".join(f"{x:>4}" for x in counts))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--claims", default=str(ROOT / "claims.json"))
    p.add_argument("--limit", type=int, help="Only use the first N claims")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key.startswith("sk-ant-") or any(x in key for x in ("...", "your", "PASTE")):
        raise SystemExit("ANTHROPIC_API_KEY is missing or still a placeholder.")

    claims = json.loads(Path(args.claims).read_text(encoding="utf-8"))
    if args.limit:
        claims = claims[: args.limit]

    # Fetch evidence up front (and cache it) so both runs see identical text.
    for page in sorted({c["page"] for c in claims}):
        fetch_wikipedia(page)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary = [f"Model: {MODEL}  effort: {args.effort}  claims: {len(claims)}  run: {stamp}"]

    for name, use_evidence in CONDITIONS.items():
        print(f"Running {name} on {len(claims)} claims...")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(lambda c: run_one(c, use_evidence, args.effort), claims))

        with open(out_dir / f"{stamp}_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        for r in rows:
            if not r["correct"]:
                print(f"   MISS #{r['id']}: gold={r['gold']} pred={r['pred']} | {r['claim']}")
        tokens_in = sum(r["input_tokens"] for r in rows)
        tokens_out = sum(r["output_tokens"] for r in rows)
        block = report(name, rows) + f"\n   tokens: {tokens_in:,} in / {tokens_out:,} out"
        print(block)
        summary.append(block)

    (out_dir / f"{stamp}_summary.txt").write_text("\n\n".join(summary) + "\n", encoding="utf-8")
    print(f"\nSaved per-claim results and summary to {out_dir}/")


if __name__ == "__main__":
    main()
