from pathlib import Path

from sysforge.agent.llm import (
    LLMError,
    chat_json,
    chat_text,
    extract_fenced_block,
    extract_json_object,
    has_llm_config,
    is_transient,
    resolve_api_key,
)
from sysforge.agent.prompts import cuda_prompt_file, json_prompt, render_prompt, system_prompt


def test_extract_fenced_block_prefers_requested_language():
    text = "```python\nprint('x')\n```\n```cuda\nextern \"C\" int x;\n```"
    assert extract_fenced_block(text, preferred_lang="cuda") == 'extern "C" int x;'


def test_extract_json_object_recovers_from_wrapped_response():
    obj = extract_json_object("before\n```json\n{\"a\": 1}\n```\nafter")
    assert obj == {"a": 1}


def test_prompt_helpers_load_and_render(tmp_path: Path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "system.txt").write_text("system", encoding="utf-8")
    (prompt_dir / "hello.txt").write_text("hello {name}", encoding="utf-8")
    assert system_prompt(prompt_dir) == "system"
    assert render_prompt(prompt_dir, "hello.txt", name="world") == "hello world"
    assert system_prompt(prompt_dir) == "system"


def test_chat_text_uses_configured_timeout(monkeypatch):
    monkeypatch.setenv("BASE_MODEL", "stub-model")
    monkeypatch.setenv("LLM_TIMEOUT_S", "17")
    captured = {}

    class FakeClient:
        def with_options(self, *, timeout):
            captured["timeout"] = timeout
            return self

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, *, model, messages, temperature):
            captured["model"] = model
            captured["messages"] = messages
            captured["temperature"] = temperature

            class Message:
                content = "ok"

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    def fake_get_client():
        return FakeClient()

    monkeypatch.setattr("sysforge.agent.llm.get_client", fake_get_client)
    out = chat_text("ping", system="sys", temperature=0.4, transient_retries=2)
    assert out == "ok"
    assert captured["model"] == "stub-model"
    assert captured["temperature"] == 0.4
    assert captured["messages"][0]["content"] == "sys"
    assert captured["messages"][1]["content"] == "ping"
    assert captured["timeout"] == 17.0


def test_chat_text_retries_without_temperature_when_backend_rejects_it(monkeypatch):
    monkeypatch.setenv("BASE_MODEL", "stub-model")
    calls = []

    class UnsupportedTemperatureError(Exception):
        pass

    class FakeClient:
        def with_options(self, *, timeout):
            return self

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            calls.append(kwargs)
            if "temperature" in kwargs:
                raise UnsupportedTemperatureError(
                    "Unsupported value: 'temperature' does not support 0 with this model. Only the default (1) value is supported."
                )

            class Message:
                content = "ok"

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    monkeypatch.setattr("sysforge.agent.llm.get_client", lambda: FakeClient())

    out = chat_text("ping", temperature=0.0, transient_retries=0)

    assert out == "ok"
    assert len(calls) == 2
    assert calls[0]["temperature"] == 0.0
    assert "temperature" not in calls[1]


def test_has_llm_config_accepts_custom_base_url_without_api_key():
    assert has_llm_config(api_key="", base_url="http://127.0.0.1:8000/v1", model="stub-model") is True
    assert has_llm_config(api_key="", base_url="", model="stub-model") is False
    assert has_llm_config(api_key="stub", base_url="", model="stub-model") is True
    assert has_llm_config(api_key="", base_url="http://127.0.0.1:8000/v1", model="") is False


def test_resolve_api_key_uses_dummy_for_custom_base_url():
    assert resolve_api_key(api_key="", base_url="http://127.0.0.1:8000/v1") == "dummy"


def test_is_transient_detects_common_transport_errors():
    assert is_transient(RuntimeError("502 bad gateway")) is True
    assert is_transient(RuntimeError("syntax problem")) is False


def test_extract_json_object_raises_for_missing_object():
    try:
        extract_json_object("not json")
    except LLMError as exc:
        assert "No JSON object found" in str(exc)
    else:
        raise AssertionError("expected LLMError")


def test_chat_json_reprompts_after_invalid_json(monkeypatch):
    calls = 0
    captured = {}

    def fake_chat_text(prompt, **kwargs):
        nonlocal calls
        calls += 1
        captured["prompt"] = prompt
        return "not json" if calls == 1 else '{"ok": true}'

    monkeypatch.setattr("sysforge.agent.llm.chat_text", fake_chat_text)
    assert chat_json("return json", retries=1) == {"ok": True}
    assert "could not be parsed as JSON" in captured["prompt"]


def test_chat_json_forwards_transient_retry_budget(monkeypatch):
    captured = {}

    def fake_chat_text(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return '{"ok": true}'

    monkeypatch.setattr("sysforge.agent.llm.chat_text", fake_chat_text)

    assert chat_json("return json", retries=0, transient_retries=0) == {"ok": True}
    assert captured["transient_retries"] == 0


def test_json_prompt_renders_and_delegates(monkeypatch, tmp_path: Path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "system.txt").write_text("system", encoding="utf-8")
    (prompt_dir / "json_system.txt").write_text("json-system", encoding="utf-8")
    (prompt_dir / "hello.txt").write_text('{{"x":"{name}"}}', encoding="utf-8")
    captured = {}

    def fake_chat_json(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("sysforge.agent.prompts.chat_json", fake_chat_json)
    assert json_prompt(prompt_dir, "hello.txt", name="world") == {"ok": True}
    assert captured["prompt"] == '{"x":"world"}'
    assert captured["system"] == "system"

    captured.clear()
    assert json_prompt(prompt_dir, "hello.txt", system_name="json_system.txt", name="world") == {"ok": True}
    assert captured["system"] == "json-system"


def test_cuda_prompt_file_loads_raw_prompt(monkeypatch, tmp_path: Path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "system.txt").write_text("system", encoding="utf-8")
    (prompt_dir / "raw.txt").write_text("raw prompt", encoding="utf-8")
    captured = {}

    def fake_chat_cuda(prompt, **kwargs):
        captured["prompt"] = prompt
        return "code"

    monkeypatch.setattr("sysforge.agent.prompts.chat_cuda", fake_chat_cuda)
    assert cuda_prompt_file(prompt_dir, "raw.txt") == "code"
    assert captured["prompt"] == "raw prompt"
