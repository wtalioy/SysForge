from __future__ import annotations

import hashlib


def source_digest(source: str, *, length: int | None = None) -> str:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return digest[:length] if length is not None else digest
