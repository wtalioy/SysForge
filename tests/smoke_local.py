"""End-to-end smoke test with LLM, nvcc, and the binary all stubbed.

Run with: python -m tests.smoke_local
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock


FAKE_CUDA_SOURCE = "// fake cuda\nint main(){return 0;}\n"


def _fake_json(prompt, *, system=None, temperature=0.2, retries=1):
    if "Extract the numeric answer" in prompt:
        return {"value": 432.0, "unit": "cycles", "confidence": 0.8,
                "reasoning": "matched RESULT line"}
    if "Diagnose the primary" in prompt or "Nsight Compute CSV" in prompt:
        return {"bottleneck": "memory_bound",
                "evidence": [{"metric": "dram__throughput", "value": "82%",
                              "why": "dominant"}],
                "recommendations": ["increase tile size"],
                "summary": "DRAM bandwidth saturated."}
    return {"source": FAKE_CUDA_SOURCE, "args": [],
            "parse_hint": "RESULT line",
            "rationale": "stubbed probe"}


def _fake_cuda(prompt, *, system=None, temperature=0.2):
    return FAKE_CUDA_SOURCE


class _FakeCR:
    def __init__(self, ok=True):
        self.ok = ok
        self.binary = Path("/bin/true")
        self.stdout = ""
        self.stderr = ""
        self.cmd = ["nvcc", "stub"]


class _FakeRR:
    def __init__(self):
        self.ok = True
        self.rc = 0
        self.stdout = "RESULT dram_latency_cycles=432.1 unit=cycles samples=1024\n"
        self.stderr = ""
        self.wallclock_s = 0.01
        self.timed_out = False


class _FakeNR:
    def __init__(self):
        self.ok = True
        self.rc = 0
        self.stdout = ('"ID","Metric Name","Metric Value"\n'
                       '"0","sm__throughput.avg.pct_of_peak_sustained_elapsed","82"\n')
        self.stderr = ""
        self.rows = []


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        tgt = Path(d) / "target"
        ws = Path(d) / "workspace"
        tgt.mkdir()
        ws.mkdir()
        (tgt / "target_spec.json").write_text(json.dumps({
            "targets": ["dram_latency_cycles",
                        "sm__throughput.avg.pct_of_peak_sustained_elapsed"],
        }))
        os.environ["TARGET_DIR"] = str(tgt)
        os.environ["WORKSPACE_DIR"] = str(ws)
        os.environ.setdefault("API_KEY", "stub")
        os.environ.setdefault("BASE_MODEL", "stub")

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

        with mock.patch("sysforge.agent.prompts.chat_json", new=_fake_json), \
             mock.patch("sysforge.agent.prompts.chat_cuda", new=_fake_cuda), \
             mock.patch("sysforge.integrations.compiler.compile_cuda", return_value=_FakeCR()), \
             mock.patch("sysforge.integrations.executor.run_binary", return_value=_FakeRR()), \
             mock.patch("sysforge.integrations.ncu.profile", return_value=_FakeNR()):
            from sysforge.main import main as agent_main
            rc = agent_main(["profiling"])

        out_path = ws / "output.json"
        assert out_path.exists(), "output.json not written"
        out = json.loads(out_path.read_text())
        assert "dram_latency_cycles" in out["hardware"], out
        hw = out["hardware"]["dram_latency_cycles"]
        assert hw["value"] in (432.0, 432.1), hw
        assert len(hw["accepted_samples"]) >= 1, hw
        assert out["analysis"]["bottleneck"] == "memory_bound", out["analysis"]
        print("SMOKE OK -> ", out_path)
        print(json.dumps({"hardware": out["hardware"],
                          "analysis_bottleneck": out["analysis"]["bottleneck"]},
                         indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
