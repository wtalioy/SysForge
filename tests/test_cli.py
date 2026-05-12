from pathlib import Path

from sysforge.cli.app import build_parser, main


def test_build_parser_has_workflow_subcommands():
    parser = build_parser()
    args = parser.parse_args(["profiling"])
    assert args.workflow == "profiling"
    args2 = parser.parse_args(["optimize-lora"])
    assert args2.workflow == "optimize-lora"


def test_cli_dispatches_selected_workflow(monkeypatch, tmp_path):
    monkeypatch.setenv("TARGET_DIR", str(tmp_path / "target"))
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))
    (tmp_path / "target").mkdir()

    captured = {}

    def fake_write_output(path, payload):
        captured["path"] = path
        captured["payload"] = payload

    class FakeWorkflow:
        name = "profiling"
        description = "fake"

        def run(self, context):
            return {"workflow": self.name, "started_at": context.started_at}

    monkeypatch.setattr("sysforge.cli.app.build_registry", lambda: {"profiling": FakeWorkflow()})
    monkeypatch.setattr("sysforge.cli.app.write_output", fake_write_output)
    rc = main(["profiling"])
    assert rc == 0
    assert captured["payload"]["workflow"] == "profiling"


def test_run_sh_invokes_optimize_lora():
    run_sh = Path(__file__).resolve().parent.parent / "run.sh"
    text = run_sh.read_text(encoding="utf-8")
    assert "python3 -m sysforge.main optimize-lora" in text


def test_live_local_optimize_lora_script_has_expected_contract_checks():
    script = Path(__file__).resolve().parent / "live_optimize_lora.sh"
    text = script.read_text(encoding="utf-8")
    assert script.exists()
    assert text.startswith("#!/usr/bin/env bash")
    assert "conda activate base" in text
    assert "BASE_MODEL is required" in text
    assert 'bash "$ROOT/run.sh"' in text
    assert 'torch.utils.cpp_extension import load' in text
    assert 'optimized_lora.cu' in text
    assert 'output.json' in text
    assert 'PYBIND11_MODULE' in text
    assert 'workflow_output.get("status") not in {"optimized", "searched", "confirmed_baseline"}' in text
    assert 'workflow_output.get("llm_enabled")' in text
    assert "PASS_OPTIMIZED" in text
    assert "PASS_SEARCHED" in text
    assert "FAIL_NOT_ENOUGH_SEARCH" in text
    assert "FAIL_WORKFLOW" in text
    assert "OPTIMIZE_LORA_MAX_FAMILY_VARIANTS" in text
    assert "OPTIMIZE_LORA_MAX_FULL_EVALS_PER_ROUND" in text
    assert "OPTIMIZE_LORA_MAX_LLM_ROUNDS" in text
    assert "OPTIMIZE_LORA_CLEAR_WINNER_SPEEDUP" in text
    assert "OPTIMIZE_LORA_FINAL_CONFIRM_WARMUP" in text
    assert "OPTIMIZE_LORA_FINAL_CONFIRM_ITERS" in text
    assert "OPTIMIZE_LORA_MIN_SEED_VARIANTS" in text
    assert "OPTIMIZE_LORA_MAX_STALLED_ROUNDS" in text
    assert "SYSFORGE_RUN_TIMEOUT" in text
    assert 'workflow_output.get("status") not in {"optimized", "searched", "confirmed_baseline"}' in text
    assert "seed_count_run" in text
    assert "mutation_count_run" in text
    assert "seed_variants_screened" in text
    assert "mutation_variants_screened" in text
    assert "winner_confirmed" in text
    assert "winner_confirmation_candidate_id" in text
    assert "compile_time_s_total" in text
    assert "variants_benchmarked" in text
    assert "skipped_steps" in text
    assert "winner kind" in text
    assert "winner confirm" in text
    assert "profiling was requested but no successful profile completed; first profiler error:" in text
    assert "profiling was requested but output.json reports profiling_used=false" in text
    assert "profiling was requested but no non-baseline candidate has a successful profile_summary" in text
    assert "TORCH_CUDA_ARCH_LIST" in text


def test_run_sh_checks_dependencies_without_reinstalling():
    run_sh = Path(__file__).resolve().parent.parent / "run.sh"
    text = run_sh.read_text(encoding="utf-8")
    assert "pip3 install" not in text
    assert "Missing Python dependency for SysForge runtime" in text
