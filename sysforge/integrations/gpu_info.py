from __future__ import annotations

import shutil
import subprocess


def weak_hint() -> dict:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {}

    query_fields = [
        "name",
        "compute_cap",
        "clocks.sm",
        "clocks.max.sm",
        "memory.total",
    ]
    cmd = [
        nvidia_smi,
        f"--query-gpu={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except Exception:
        return {}
    if result.returncode != 0:
        return {}

    rows = []
    seen = set()
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(query_fields):
            continue
        row = dict(zip(query_fields, parts))
        rows.append(row)
        key = tuple(parts)
        if key not in seen:
            seen.add(key)

    unique_rows = []
    added = set()
    for row in rows:
        key = tuple(row[field] for field in query_fields)
        if key in added:
            continue
        added.add(key)
        unique_rows.append(row)
    return {"nvidia_smi": rows, "nvidia_smi_unique": unique_rows}
