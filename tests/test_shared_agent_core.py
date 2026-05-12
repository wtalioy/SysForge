from pathlib import Path

from sysforge.agent.base import BaseAgent, SearchAgent, StoppingPolicy
from sysforge.runtime import build_runtime_context, load_config


def _context(monkeypatch, tmp_path: Path, *, api_key: str = "", base_model: str = ""):
    target_dir = tmp_path / "target"
    workspace_dir = tmp_path / "workspace"
    target_dir.mkdir(parents=True)
    workspace_dir.mkdir(parents=True)
    (target_dir / "target_spec.json").write_text('{"targets":[]}', encoding="utf-8")
    monkeypatch.setenv("TARGET_DIR", str(target_dir))
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))
    monkeypatch.setenv("API_KEY", api_key)
    monkeypatch.setenv("BASE_MODEL", base_model)
    return build_runtime_context(load_config())


def test_base_agent_tracks_trace(monkeypatch, tmp_path: Path):
    agent = BaseAgent(_context(monkeypatch, tmp_path))

    agent.record_trace(action="started", detail="unit-test")
    assert agent.trace == [{"action": "started", "detail": "unit-test"}]

def test_search_agent_updates_state_and_stopping_policy(monkeypatch, tmp_path: Path):
    policy = StoppingPolicy(max_stalled_rounds=2)
    agent = SearchAgent(_context(monkeypatch, tmp_path), stop_policy=policy)

    first_round = agent.begin_round("tile")
    agent.finish_round(improved=False)
    second_round = agent.begin_round("tile")
    agent.finish_round(improved=False)
    decision = agent.stop_decision()

    assert first_round == 1
    assert second_round == 2
    assert decision.should_stop is True
    assert decision.reason == "stalled_rounds"
    assert agent.state.rounds_run == 2
    assert agent.trace == [
        {"action": "round_started", "round_index": 1, "family_name": "tile"},
        {"action": "round_started", "round_index": 2, "family_name": "tile"},
    ]
