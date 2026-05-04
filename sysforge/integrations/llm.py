from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any


class LLMError(RuntimeError):
    pass


_CLIENT = None
_TRANSIENT_MARKERS = (
    "connection", "timeout", "timed out", "temporarily unavailable",
    "rate limit", "429", "502", "503", "504", "bad gateway", "reset by peer",
    "proxy", "ssl", "eof", "incomplete read",
)
_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n?(.*?)```", re.DOTALL)


def _is_transient(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "connection" in name or "timeout" in name or "apiconnection" in name:
        return True
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def _client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    from openai import OpenAI
    api_key = os.environ.get("API_KEY", "")
    base_url = os.environ.get("BASE_URL", "") or None
    if not api_key:
        raise LLMError("API_KEY is not set")
    _CLIENT = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=float(os.environ.get("LLM_TIMEOUT_S", "120")),
        max_retries=0,
    )
    return _CLIENT


def chat_text(prompt: str, *, system: str | None = None, temperature: float = 0.2,
              transient_retries: int = 3, deadline_s: float | None = None) -> str:
    model = os.environ.get("BASE_MODEL", "")
    if not model:
        raise LLMError("BASE_MODEL is not set")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    started = time.monotonic()

    def time_left() -> float:
        if deadline_s is None:
            return float("inf")
        return deadline_s - (time.monotonic() - started)

    last_exc: Exception | None = None
    for attempt in range(transient_retries + 1):
        if time_left() <= 0:
            break
        per_call_timeout = float(os.environ.get("LLM_TIMEOUT_S", "90"))
        if deadline_s is not None:
            per_call_timeout = min(per_call_timeout, max(5.0, time_left()))
        try:
            response = _client().with_options(timeout=per_call_timeout).chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_transient(exc) or attempt == transient_retries:
                break
            backoff = min(15.0, 1.0 * (2 ** attempt)) + random.uniform(0, 0.5)
            if backoff >= time_left():
                break
            time.sleep(backoff)
    raise LLMError(
        f"LLM call failed after {attempt + 1} attempt(s): {type(last_exc).__name__}: {last_exc}"
    )


def extract_fenced_block(text: str, preferred_lang: str | None = None) -> str:
    if preferred_lang:
        pattern = re.compile(
            rf"```{re.escape(preferred_lang)}\n?(.*?)```",
            re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_json_object(text: str) -> Any:
    block = extract_fenced_block(text, preferred_lang="json")
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        pass
    start = block.find("{")
    end = block.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = block[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Could not parse JSON object: {exc}; snippet={candidate[:200]}")
    raise LLMError(f"No JSON object found in response: {text[:200]}")


def chat_json(prompt: str, *, system: str | None = None,
              temperature: float = 0.2, retries: int = 1,
              deadline_s: float | None = None) -> Any:
    last_err: Exception | None = None
    current_prompt = prompt
    started = time.monotonic()
    for _attempt in range(retries + 1):
        remaining = None if deadline_s is None else max(0.0, deadline_s - (time.monotonic() - started))
        if remaining == 0.0:
            break
        text = chat_text(current_prompt, system=system, temperature=temperature, deadline_s=remaining)
        try:
            return extract_json_object(text)
        except LLMError as exc:
            last_err = exc
            current_prompt = (
                prompt + "\n\nYour previous response could not be parsed as JSON: "
                f"{exc}\nReturn ONLY a valid JSON object, no prose, no fences."
            )
    raise LLMError(f"LLM failed to produce valid JSON after {retries + 1} attempts: {last_err}")


def chat_cuda(prompt: str, *, system: str | None = None, temperature: float = 0.2,
              deadline_s: float | None = None) -> str:
    text = chat_text(prompt, system=system, temperature=temperature, deadline_s=deadline_s)
    code = extract_fenced_block(text, preferred_lang="cuda")
    if not code.strip():
        code = extract_fenced_block(text, preferred_lang="cpp")
    if not code.strip():
        code = text.strip()
    return code
