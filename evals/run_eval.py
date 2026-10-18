"""Run the agent eval suite.

    python -m evals.run_eval --mode fallback        # no API key needed
    python -m evals.run_eval --mode llm             # uses OPENAI_API_KEY / OPENAI_MODEL / OPENAI_BASE_URL
    python -m evals.run_eval --mode both --case cmp_months

Each case runs against a fresh in-memory database seeded with the same
deterministic history and a fixed "today", so results are reproducible apart
from LLM nondeterminism. Scores:
  - tool selection: required tools were called and forbidden tools were not
  - answer correctness: every ground-truth number appears in the reply and DB checks pass
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from src.agent.agent import FinanceAgent
from src.agent.llm import OpenAIChatClient
from src.agent.tools import ToolRegistry
from src.services.conversation_store import ConversationStore
from src.services.db import init_db, make_engine, make_session_factory
from src.services.insight_generator import SimpleInsightGenerator
from src.services.models import User
from src.services.sql_transaction_service import SQLTransactionService

from .cases import CASES, Case
from .dataset import EVAL_TODAY, seed

ROOT = Path(__file__).resolve().parent.parent

# USD per 1M tokens (input, output). Estimates; update for your provider.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


@dataclass
class CaseResult:
    id: str
    tag: str
    mode: str
    passed: bool
    tools_ok: bool
    answer_ok: bool
    tools_called: list[str]
    missing_numbers: list[float]
    notes: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reply: str = ""


def extract_numbers(text: str) -> list[float]:
    values = []
    for raw in _NUMBER.findall(text):
        try:
            values.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return values


def numbers_present(reply: str, expected: list[float], tolerance: float) -> list[float]:
    found = extract_numbers(reply)
    return [e for e in expected if not any(abs(abs(f) - abs(e)) <= max(tolerance, 0.011) for f in found)]


def _build(mode: str):
    engine = make_engine("sqlite://")
    init_db(engine)
    factory = make_session_factory(engine)
    tx = SQLTransactionService(factory, today=lambda: EVAL_TODAY)
    with factory() as session:
        user, other = User(username="eval", password_hash="x"), User(username="other", password_hash="x")
        session.add_all([user, other])
        session.commit()
        uid, other_id = user.id, other.id
    seed(tx, uid, other_id)

    llm = _llm_client() if mode == "llm" else None
    agent = FinanceAgent(
        ToolRegistry.default(tx, SimpleInsightGenerator("₹")),
        ConversationStore(factory),
        llm,
        currency="₹",
        today=lambda: EVAL_TODAY,
    )
    return engine, tx, agent, uid


async def run_case(case: Case, mode: str) -> CaseResult:
    engine, tx, agent, uid = _build(mode)
    expected = case.numbers(tx, uid) if case.numbers else []  # ground truth before any writes
    for turn in case.turns[:-1]:
        await agent.respond(turn, uid)
    result = await agent.respond(case.turns[-1], uid)

    called = [c["name"] for c in result.tool_calls]
    notes: list[str] = []
    tools_ok = True
    if case.tools_any and not set(called) & case.tools_any:
        tools_ok = False
        notes.append(f"expected one of {sorted(case.tools_any)}")
    if case.no_tools and called:
        tools_ok = False
        notes.append("expected no tool calls")
    if set(called) & case.forbid_tools:
        tools_ok = False
        notes.append(f"called forbidden {sorted(set(called) & case.forbid_tools)}")
    if result.mode != mode:
        tools_ok = False
        notes.append(f"ran in {result.mode} mode (LLM unavailable)")

    missing = numbers_present(result.reply, expected, case.number_tolerance)
    answer_ok = not missing
    if missing:
        notes.append(f"missing numbers {missing}")
    if case.db_check and not case.db_check(tx, uid):
        answer_ok = False
        notes.append("database check failed")
    leaked = [t for t in case.forbid_text if t in result.reply]
    if leaked:
        answer_ok = False
        notes.append(f"forbidden text in reply {leaked}")
    engine.dispose()

    return CaseResult(
        id=case.id,
        tag=case.tag,
        mode=mode,
        passed=tools_ok and answer_ok,
        tools_ok=tools_ok,
        answer_ok=answer_ok,
        tools_called=called,
        missing_numbers=missing,
        notes=notes,
        latency_ms=result.latency_ms,
        steps=result.steps,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        reply=result.reply,
    )


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summarize(results: list[CaseResult], model: str | None) -> dict:
    n = len(results)
    latencies = [r.latency_ms for r in results]
    summary = {
        "cases": n,
        "pass_rate": sum(r.passed for r in results) / n,
        "tool_selection_accuracy": sum(r.tools_ok for r in results) / n,
        "answer_accuracy": sum(r.answer_ok for r in results) / n,
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(_pct(latencies, 0.95), 1),
        "avg_steps": round(sum(r.steps for r in results) / n, 2),
        "by_tag": {},
    }
    for tag in sorted({r.tag for r in results}):
        group = [r for r in results if r.tag == tag]
        summary["by_tag"][tag] = f"{sum(r.passed for r in group)}/{len(group)}"
    if model:
        prompt = sum(r.prompt_tokens for r in results)
        completion = sum(r.completion_tokens for r in results)
        summary["model"] = model
        summary["avg_tokens_per_case"] = round((prompt + completion) / n)
        if model in PRICES:
            pin, pout = PRICES[model]
            summary["est_cost_per_case_usd"] = round((prompt * pin + completion * pout) / 1e6 / n, 6)
    return summary


def render_markdown(mode: str, summary: dict, results: list[CaseResult]) -> str:
    lines = [f"### Mode: `{mode}`" + (f" (model `{summary['model']}`)" if "model" in summary else ""), ""]
    lines += [
        "| Metric | Value |",
        "|---|---|",
        f"| Cases | {summary['cases']} |",
        f"| Pass rate | {summary['pass_rate']:.0%} |",
        f"| Tool selection accuracy | {summary['tool_selection_accuracy']:.0%} |",
        f"| Answer correctness (numbers match DB) | {summary['answer_accuracy']:.0%} |",
        f"| Latency p50 / p95 | {summary['latency_p50_ms']:.0f} ms / {summary['latency_p95_ms']:.0f} ms |",
        f"| Avg model calls per question | {summary['avg_steps']} |",
    ]
    if "avg_tokens_per_case" in summary:
        lines.append(f"| Avg tokens per question | {summary['avg_tokens_per_case']:,} |")
    if "est_cost_per_case_usd" in summary:
        lines.append(f"| Est. cost per question | ${summary['est_cost_per_case_usd']:.5f} |")
    lines += ["", "By category: " + ", ".join(f"{k} {v}" for k, v in summary["by_tag"].items()), ""]
    failures = [r for r in results if not r.passed]
    if failures:
        lines += ["Failures:", ""]
        lines += [f"- `{r.id}`: {'; '.join(r.notes)} (tools: {r.tools_called or 'none'})" for r in failures]
        lines.append("")
    return "\n".join(lines)


def _llm_client() -> OpenAIChatClient:
    return OpenAIChatClient(
        os.environ["OPENAI_API_KEY"],
        os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )


async def llm_preflight() -> bool:
    """One tiny request so a bad key / no credits fails fast instead of 30 silent fallbacks."""
    try:
        async for _ in _llm_client().stream([{"role": "user", "content": "Reply with OK."}], []):
            pass
    except Exception as exc:
        print(f"LLM preflight failed, skipping llm mode: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False
    return True


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["llm", "fallback", "both"], default="both")
    parser.add_argument("--case", action="append", help="Run only these case ids")
    parser.add_argument("--write-results", action="store_true", help="Update evals/RESULTS.md")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print every reply")
    args = parser.parse_args()
    # Agent logs full tracebacks when the LLM fails; the per-case notes are enough here.
    logging.getLogger("src.agent.agent").setLevel(logging.CRITICAL)

    modes = ["fallback", "llm"] if args.mode == "both" else [args.mode]
    if "llm" in modes and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; skipping llm mode.", file=sys.stderr)
        modes.remove("llm")
    if "llm" in modes and not await llm_preflight():
        modes.remove("llm")
    cases = [c for c in CASES if not args.case or c.id in args.case]

    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    sections = []
    for mode in modes:
        results = []
        for case in cases:
            res = await run_case(case, mode)
            results.append(res)
            mark = "PASS" if res.passed else "FAIL"
            print(f"[{mode:8}] {mark} {case.id:22} {res.latency_ms:8.0f} ms  tools={res.tools_called}"
                  + (f"  {'; '.join(res.notes)}" if res.notes else ""))
            if args.verbose or not res.passed:
                print("           reply:", res.reply.replace("\n", " ")[:400])
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini") if mode == "llm" else None
        summary = summarize(results, model)
        print(json.dumps(summary, indent=2))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        (report_dir / f"eval_{mode}_{stamp}.json").write_text(
            json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        sections.append(render_markdown(mode, summary, results))

    if args.write_results and sections and not args.case:
        header = (
            "# Eval results\n\n"
            f"Generated by `python -m evals.run_eval --write-results` on {datetime.now():%Y-%m-%d}. "
            f"{len(cases)} cases, synthetic history of 3.5 months, fixed date {EVAL_TODAY}.\n"
            "A case passes when the right tools were called (and forbidden ones were not) and every "
            "ground-truth number computed from the database appears in the reply.\n\n"
        )
        (ROOT / "evals" / "RESULTS.md").write_text(header + "\n".join(sections), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
