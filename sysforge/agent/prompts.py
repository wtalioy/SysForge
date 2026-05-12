from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

from .llm import chat_cuda, chat_json


_OPEN_BRACE_SENTINEL = "\uFFF0"
_CLOSE_BRACE_SENTINEL = "\uFFF1"
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@lru_cache(maxsize=None)
def load_prompt(prompt_dir: Path | str, prompt_name: str) -> str:
    return (Path(prompt_dir) / prompt_name).read_text(encoding="utf-8")


def render_prompt(prompt_dir: Path, prompt_name: str, **kwargs) -> str:
    template = load_prompt(prompt_dir, prompt_name)
    template = template.replace("{{", _OPEN_BRACE_SENTINEL).replace("}}", _CLOSE_BRACE_SENTINEL)
    rendered = _PLACEHOLDER_RE.sub(lambda match: str(kwargs.get(match.group(1), match.group(0))), template)
    return rendered.replace(_OPEN_BRACE_SENTINEL, "{").replace(_CLOSE_BRACE_SENTINEL, "}")


@lru_cache(maxsize=None)
def system_prompt(prompt_dir: Path | str, system_name: str = "system.txt") -> str:
    return load_prompt(prompt_dir, system_name)


def json_prompt(
    prompt_dir: Path,
    prompt_name: str,
    *,
    system_name: str = "system.txt",
    temperature: float = 0.2,
    retries: int = 1,
    transient_retries: int = 3,
    **kwargs,
):
    return chat_json(
        render_prompt(prompt_dir, prompt_name, **kwargs),
        system=system_prompt(prompt_dir, system_name=system_name),
        temperature=temperature,
        retries=retries,
        transient_retries=transient_retries,
    )


def cuda_prompt_file(
    prompt_dir: Path,
    prompt_name: str,
    *,
    system_name: str = "system.txt",
    temperature: float = 0.2,
) -> str:
    return chat_cuda(
        load_prompt(prompt_dir, prompt_name),
        system=system_prompt(prompt_dir, system_name=system_name),
        temperature=temperature,
    )
