from types import SimpleNamespace

from sysforge.integrations import compiler, gpu_info, ncu


def test_compile_cuda_reports_missing_nvcc_without_running_subprocess(monkeypatch, tmp_path):
    monkeypatch.setattr(compiler, "find_nvcc", lambda: None)

    result = compiler.compile_cuda(tmp_path / "probe.cu", tmp_path / "probe")

    assert result.ok is False
    assert result.binary is None
    assert result.stderr == "nvcc not found on PATH"
    assert result.cmd == []


def test_compile_cuda_builds_expected_command_and_handles_success(monkeypatch, tmp_path):
    source = tmp_path / "probe.cu"
    binary = tmp_path / "build" / "probe"
    source.write_text("int main(){return 0;}", encoding="utf-8")
    captured = {}

    def fake_run(cmd, *, capture_output, text, timeout):
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        binary.write_text("binary", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(compiler, "find_nvcc", lambda: "/usr/bin/nvcc")
    monkeypatch.setattr(compiler.subprocess, "run", fake_run)

    result = compiler.compile_cuda(source, binary, arch="sm_90", timeout_s=7, extra_flags=["--use_fast_math"])

    assert result.ok is True
    assert result.binary == binary
    assert captured["timeout"] == 7
    assert captured["cmd"] == [
        "/usr/bin/nvcc",
        str(source),
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "-o",
        str(binary),
        "-arch",
        "sm_90",
        "--use_fast_math",
    ]


def test_ncu_profile_command_parses_csv_and_reports_missing_tool(monkeypatch):
    monkeypatch.setattr(ncu, "find_ncu", lambda: None)
    missing = ncu.profile_command(["/bin/true"], ["sm__throughput"])
    assert missing.ok is False
    assert missing.stderr == "ncu not found on PATH"

    def fake_run(cmd, *, capture_output, text, timeout):
        return SimpleNamespace(
            returncode=0,
            stdout='header\n"ID","Metric Name","Metric Value"\n"0","sm__throughput","82"\n',
            stderr="",
        )

    monkeypatch.setattr(ncu, "find_ncu", lambda: "/usr/bin/ncu")
    monkeypatch.setattr(ncu.subprocess, "run", fake_run)

    result = ncu.profile_command(["/bin/true"], ["sm__throughput"], timeout_s=3)

    assert result.ok is True
    assert result.rows == [{"ID": "0", "Metric Name": "sm__throughput", "Metric Value": "82"}]


def test_weak_hint_deduplicates_nvidia_smi_rows(monkeypatch):
    monkeypatch.setattr(gpu_info.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        gpu_info.subprocess,
        "run",
        lambda cmd, *, capture_output, text, timeout: SimpleNamespace(
            returncode=0,
            stdout="GPU, 9.0, 1000, 1500, 8192\nGPU, 9.0, 1000, 1500, 8192\n",
        ),
    )

    hints = gpu_info.weak_hint()

    assert len(hints["nvidia_smi"]) == 2
    assert hints["nvidia_smi_unique"] == [
        {
            "name": "GPU",
            "compute_cap": "9.0",
            "clocks.sm": "1000",
            "clocks.max.sm": "1500",
            "memory.total": "8192",
        }
    ]
