from pathlib import Path

from sysforge.integrations.executor import run_binary


def test_timeout(tmp_path: Path):
    # /bin/sleep is ubiquitous on linux
    sleep = Path("/bin/sleep")
    if not sleep.exists():
        return
    r = run_binary(sleep, ["5"], timeout_s=0.1)
    assert r.timed_out
    assert not r.ok


def test_exit_code(tmp_path: Path):
    false_bin = Path("/bin/false")
    if not false_bin.exists():
        return
    r = run_binary(false_bin, timeout_s=5)
    assert r.rc != 0
    assert not r.ok
