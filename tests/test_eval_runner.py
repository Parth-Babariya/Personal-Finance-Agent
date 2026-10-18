"""Eval runner checkpointing: a run interrupted by a quota resumes without mixing results."""

import sys

import pytest

from evals import run_eval
from evals.cases import CASES


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "CHECKPOINT", tmp_path / "checkpoint.json")
    monkeypatch.setattr(run_eval, "ROOT", tmp_path)  # reports and fingerprint stay inside tmp_path
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "model-a")
    monkeypatch.setattr(run_eval, "load_dotenv", lambda *a, **k: None)

    async def preflight():
        return True

    monkeypatch.setattr(run_eval, "llm_preflight", preflight)
    return tmp_path


def _fake_run_case(fail_at: str | None, calls: list[str]):
    async def fake(case, mode):
        calls.append(case.id)
        notes = ["ran in fallback mode (LLM unavailable)"] if case.id == fail_at else []
        return run_eval.CaseResult(id=case.id, tag=case.tag, mode=mode, passed=not notes, tools_ok=not notes,
                                   answer_ok=True, tools_called=[], missing_numbers=[], notes=notes)
    return fake


async def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["run_eval", "--mode", "llm", *argv])
    return await run_eval.main()


async def test_interrupted_run_resumes_only_remaining_cases(runner, monkeypatch):
    third = CASES[2].id
    first_calls: list[str] = []
    monkeypatch.setattr(run_eval, "run_case", _fake_run_case(fail_at=third, calls=first_calls))
    await _main(monkeypatch)
    assert first_calls == [c.id for c in CASES[:3]]
    assert run_eval.CHECKPOINT.exists()  # two cases saved, the failed one is not

    second_calls: list[str] = []
    monkeypatch.setattr(run_eval, "run_case", _fake_run_case(fail_at=None, calls=second_calls))
    await _main(monkeypatch, "--resume")
    assert second_calls == [c.id for c in CASES[2:]]  # resumes at the case that failed
    assert not run_eval.CHECKPOINT.exists()  # removed once the run completes


async def test_resume_refuses_a_checkpoint_from_another_model(runner, monkeypatch, capsys):
    monkeypatch.setattr(run_eval, "run_case", _fake_run_case(fail_at=CASES[1].id, calls=[]))
    await _main(monkeypatch)

    monkeypatch.setenv("OPENAI_MODEL", "model-b")
    calls: list[str] = []
    monkeypatch.setattr(run_eval, "run_case", _fake_run_case(fail_at=None, calls=calls))
    await _main(monkeypatch, "--resume")
    assert calls == [] and "different model" in capsys.readouterr().err


async def test_fresh_run_ignores_old_checkpoint(runner, monkeypatch):
    monkeypatch.setattr(run_eval, "run_case", _fake_run_case(fail_at=CASES[1].id, calls=[]))
    await _main(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(run_eval, "run_case", _fake_run_case(fail_at=None, calls=calls))
    await _main(monkeypatch)  # no --resume: starts over
    assert calls == [c.id for c in CASES]
