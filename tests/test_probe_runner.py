from sysforge.profiling.models import ProbeOutcome
from sysforge.profiling.probe_runner import ProbeCoordinator
from sysforge.profiling.targets import strategy_for


def test_extract_numeric_result_prefers_regex_fallback_when_llm_has_no_value():
    coordinator = ProbeCoordinator.__new__(ProbeCoordinator)
    spec = strategy_for("dram_latency_cycles")
    extracted, numeric = coordinator._extract_numeric_result(
        "dram_latency_cycles",
        spec,
        "RESULT dram_latency_cycles=432.5 unit=cycles\n",
        {"value": None, "unit": "cycles", "confidence": 0.0, "reasoning": "none"},
    )
    assert extracted["value"] == 432.5
    assert numeric == 432.5


def test_finalize_outcome_uses_median_and_confidence_growth():
    coordinator = ProbeCoordinator.__new__(ProbeCoordinator)
    outcome = ProbeOutcome(
        target="dram_latency_cycles",
        unit="cycles",
        value=None,
        confidence=0.0,
        reasoning="",
        accepted_samples=[
            {"version": 1, "value": 500.0, "confidence": 0.6, "reasoning": "a"},
            {"version": 2, "value": 400.0, "confidence": 0.8, "reasoning": "b"},
            {"version": 3, "value": 450.0, "confidence": 0.8, "reasoning": "c"},
        ],
    )
    spec = strategy_for("dram_latency_cycles")
    coordinator._finalize_outcome(
        outcome,
        "dram_latency_cycles",
        spec,
        "RESULT dram_latency_cycles=450 unit=cycles\n",
        {"confidence": 0.8, "reasoning": "matched"},
        450.0,
    )
    assert outcome.value == 450.0
    assert outcome.unit == "cycles"
    assert outcome.confidence >= 0.99 or outcome.confidence > 0.8
    assert "Median of 3 accepted sample(s)" in outcome.reasoning
