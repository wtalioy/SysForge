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
    "connection",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "rate limit",
    "429",
    "502",
    "503",
    "504",
    "bad gateway",
    "reset by peer",
    "proxy",
    "ssl",
    "eof",
    "incomplete read",
)
_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n?(.*?)```", re.DOTALL)


def is_transient(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "connection" in name or "timeout" in name or "apiconnection" in name:
        return True
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def is_temperature_unsupported(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "temperature" in message
        and ("unsupported value" in message or "does not support" in message)
    )


def has_llm_config(*, api_key: str | None = None, base_url: str | None = None, model: str | None = None) -> bool:
    model = (model if model is not None else os.environ.get("BASE_MODEL", "")).strip()
    api_key = (api_key if api_key is not None else os.environ.get("API_KEY", "")).strip()
    base_url = (base_url if base_url is not None else os.environ.get("BASE_URL", "")).strip()
    if not model:
        return False
    return bool(api_key or base_url)


def resolve_api_key(*, api_key: str | None = None, base_url: str | None = None) -> str:
    api_key = (api_key if api_key is not None else os.environ.get("API_KEY", "")).strip()
    base_url = (base_url if base_url is not None else os.environ.get("BASE_URL", "")).strip()
    if api_key:
        return api_key
    if base_url:
        return "dummy"
    raise LLMError("API_KEY is not set")


def get_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    from openai import OpenAI

    api_key = resolve_api_key()
    base_url = os.environ.get("BASE_URL", "") or None
    _CLIENT = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=float(os.environ.get("LLM_TIMEOUT_S", "120")),
        max_retries=0,
    )
    return _CLIENT

def chat_text(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
    transient_retries: int = 3,
) -> str:
    model = os.environ.get("BASE_MODEL", "")
    if not model:
        raise LLMError("BASE_MODEL is not set")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_exc: Exception | None = None
    attempt = 0
    allow_temperature = True
    for attempt in range(transient_retries + 1):
        per_call_timeout = float(os.environ.get("LLM_TIMEOUT_S", "90"))
        while True:
            try:
                request_kwargs = {
                    "model": model,
                    "messages": messages,
                }
                if allow_temperature:
                    request_kwargs["temperature"] = temperature
                response = get_client().with_options(timeout=per_call_timeout).chat.completions.create(
                    **request_kwargs,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if allow_temperature and is_temperature_unsupported(exc):
                    allow_temperature = False
                    continue
                if not is_transient(exc) or attempt == transient_retries:
                    break
                backoff = min(15.0, 1.0 * (2**attempt)) + random.uniform(0, 0.5)
                time.sleep(backoff)
                break

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
    starts = [index for index, char in enumerate(block) if char == "{"][:64]
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for end in range(start, len(block)):
            char = block[end]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = block[start : end + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    start = block.find("{")
    end = block.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = block[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Could not parse JSON object: {exc}; snippet={candidate[:200]}")
    raise LLMError(f"No JSON object found in response: {text[:200]}")


def chat_json(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
    retries: int = 1,
    transient_retries: int = 3,
) -> Any:
    last_err: Exception | None = None
    current_prompt = prompt
    for _attempt in range(retries + 1):
        text = chat_text(
            current_prompt,
            system=system,
            temperature=temperature,
            transient_retries=transient_retries,
        )
        try:
            return extract_json_object(text)
        except LLMError as exc:
            last_err = exc
            current_prompt = (
                prompt
                + "\n\nYour previous response could not be parsed as JSON: "
                + f"{exc}\nReturn ONLY a valid JSON object, no prose, no fences."
            )
    raise LLMError(f"LLM failed to produce valid JSON after {retries + 1} attempts: {last_err}")


def chat_cuda(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
) -> str:
    text = chat_text(
        prompt,
        system=system,
        temperature=temperature,
    )
    code = extract_fenced_block(text, preferred_lang="cuda")
    if not code.strip():
        code = extract_fenced_block(text, preferred_lang="cpp")
    if not code.strip():
        code = text.strip()
    return code
