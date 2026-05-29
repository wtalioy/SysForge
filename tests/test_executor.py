from pathlib import Path

from sysforge.integrations.executor import run_binary


def test_run_binary_reports_timeout_and_nonzero_exit():
    # /bin/sleep is ubiquitous on linux
    sleep = Path("/bin/sleep")
    if sleep.exists():
        r = run_binary(sleep, ["5"], timeout_s=0.1)
        assert r.timed_out
        assert not r.ok

    false_bin = Path("/bin/false")
    if false_bin.exists():
        r = run_binary(false_bin, timeout_s=5)
        assert r.rc != 0
        assert not r.ok
