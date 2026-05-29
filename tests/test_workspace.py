from sysforge.integrations.workspace import Workspace, safe_name


def test_workspace_sanitizes_names_and_creates_probe_paths(tmp_path):
    workspace = Workspace(
        probes_dir=tmp_path / "probes",
        build_dir=tmp_path / "build",
        logs_dir=tmp_path / "logs",
    )

    assert safe_name("dram latency/cycles") == "dram_latency_cycles"
    assert safe_name("!!!") == "probe"

    source_path = workspace.write_source("dram latency/cycles", 2, "int main(){}")
    assert source_path == tmp_path / "probes" / "dram_latency_cycles_v2.cu"
    assert source_path.read_text(encoding="utf-8") == "int main(){}"
    assert workspace.binary_path("dram latency/cycles", 2) == tmp_path / "build" / "dram_latency_cycles_v2"


def test_workspace_appends_logs_and_builds_candidate_paths(tmp_path):
    workspace = Workspace(
        probes_dir=tmp_path / "probes",
        build_dir=tmp_path / "build",
        logs_dir=tmp_path / "logs",
    )

    first_log = workspace.write_log("candidate/hash", "first")
    second_log = workspace.write_log("candidate/hash", "second")

    assert first_log == second_log
    assert first_log.read_text(encoding="utf-8") == "first\nsecond\n"
    assert workspace.candidate_source_path("a/b") == tmp_path / "probes" / "candidate_a_b.cu"
    assert workspace.candidate_build_dir("a/b") == tmp_path / "build" / "candidate_a_b"
    assert workspace.candidate_log_path("a/b") == tmp_path / "logs" / "candidate_a_b.log"
