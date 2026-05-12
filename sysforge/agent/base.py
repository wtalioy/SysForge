from __future__ import annotations

from dataclasses import dataclass

from ..runtime import RuntimeContext


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str = ""


@dataclass
class AgentState:
    rounds_run: int = 0
    stalled_rounds: int = 0

@dataclass(frozen=True)
class StoppingPolicy:
    max_stalled_rounds: int = 3

    def evaluate(self, *, state: AgentState) -> StopDecision:
        if state.stalled_rounds >= self.max_stalled_rounds:
            return StopDecision(True, "stalled_rounds")
        return StopDecision(False, "")


@dataclass
class BaseAgent:
    context: RuntimeContext

    def __post_init__(self) -> None:
        self.trace: list[dict] = []

    def record_trace(self, **payload) -> None:
        self.trace.append(payload)


class SearchAgent(BaseAgent):
    def __init__(self, context: RuntimeContext, *, stop_policy: StoppingPolicy | None = None) -> None:
        super().__init__(context=context)
        self.stop_policy = stop_policy or StoppingPolicy()
        self.state = AgentState()

    def begin_round(self, family_name: str) -> int:
        self.state.rounds_run += 1
        self.record_trace(action="round_started", round_index=self.state.rounds_run, family_name=family_name)
        return self.state.rounds_run

    def finish_round(self, *, improved: bool) -> None:
        if improved:
            self.state.stalled_rounds = 0
        else:
            self.state.stalled_rounds += 1

    def stop_decision(self) -> StopDecision:
        return self.stop_policy.evaluate(state=self.state)
