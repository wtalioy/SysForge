from sysforge.workflows.optimize_lora.models import CandidateFamilyDraft
from sysforge.workflows.optimize_lora import prompting


def test_generate_candidate_family_parses_source_and_parameters(monkeypatch):
    def fake_json_prompt(*args, **kwargs):
        return {
            "family_name": "rank16_tile_family",
            "rationale": "Search a custom low-rank kernel family.",
            "expected_bottleneck": "memory_bound",
            "source_template": "auto out = torch::matmul(W, X);\nreturn out; // {{BLOCK_X}}",
            "parameters": [
                {"name": "BLOCK_X", "values": [16, 24, 32], "default": 32, "description": "tile width"},
            ],
        }

    monkeypatch.setattr("sysforge.workflows.optimize_lora.prompting.json_prompt", fake_json_prompt)
    family = prompting.generate_candidate_family(
        env_summary="gpu=RTX3090",
        baseline_source="baseline",
        tried_family_names=["old_family"],
        recent_body_history="(none)",
        recent_plan_summary="(none)",
        recent_body_values=[],
    )
    assert isinstance(family, CandidateFamilyDraft)
    assert family.family_name == "rank16_tile_family"
    assert family.parameters[0].name == "BLOCK_X"
    assert family.parameters[0].values == (16, 24, 32)


def test_generate_candidate_family_parses_concrete_variants(monkeypatch):
    def fake_json_prompt(*args, **kwargs):
        return {
            "family_name": "explicit_variants_family",
            "rationale": "Choose a curated candidate set directly.",
            "expected_bottleneck": "launch_bound",
            "source_template": "{{FORWARD_BODY}}",
            "concrete_variants": [
                {"parameter_values": {"FORWARD_BODY": "auto out = torch::matmul(W, X); return out;"}},
                {"parameter_values": {"FORWARD_BODY": "auto out = torch::mm(W, X); return out;"}},
            ],
        }

    monkeypatch.setattr("sysforge.workflows.optimize_lora.prompting.json_prompt", fake_json_prompt)
    family = prompting.generate_candidate_family(
        env_summary="gpu=RTX3090",
        baseline_source="baseline",
        tried_family_names=[],
        recent_body_history="(none)",
        recent_plan_summary="(none)",
        recent_body_values=[],
        min_distinct_variants=2,
    )
    assert family.family_name == "explicit_variants_family"
    assert family.parameters == ()
    assert len(family.concrete_variants) == 2
    assert family.concrete_variants[1]["FORWARD_BODY"] == "auto out = torch::mm(W, X); return out;"


def test_generate_candidate_family_retries_when_parameters_do_not_affect_source(monkeypatch):
    responses = iter(
        [
            {
                "family_name": "bad_family",
                "rationale": "broken",
                "expected_bottleneck": "memory_bound",
                "source_template": "auto out = torch::matmul(W, X);\nreturn out;",
                "parameters": [{"name": "BLOCK_X", "values": [16, 32], "default": 32}],
            },
            {
                "family_name": "good_family",
                "rationale": "fixed",
                "expected_bottleneck": "memory_bound",
                "source_template": "auto out = torch::matmul(W, X);\nreturn out; // {{BLOCK_X}} {{USE_OUT}}",
                "parameters": [
                    {"name": "BLOCK_X", "values": [16, 32], "default": 32},
                    {"name": "USE_OUT", "values": [True, False], "default": True},
                ],
            },
        ]
    )

    def fake_json_prompt(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr("sysforge.workflows.optimize_lora.prompting.json_prompt", fake_json_prompt)
    family = prompting.generate_candidate_family(
        env_summary="gpu=RTX3090",
        baseline_source="baseline",
        tried_family_names=[],
        recent_body_history="(none)",
        recent_plan_summary="(none)",
        recent_body_values=[],
    )
    assert family.family_name == "good_family"
    assert len(family.parameters) == 2


def test_generate_candidate_family_retries_when_body_contains_top_level_source(monkeypatch):
    responses = iter(
        [
            {
                "family_name": "bad_partial_source",
                "rationale": "broken",
                "expected_bottleneck": "mixed",
                "source_template": "#include <torch/extension.h>\nauto out = torch::matmul(W, X);\nreturn out;",
                "parameters": [{"name": "BLOCK_X", "values": [16, 32], "default": 32}],
            },
            {
                "family_name": "good_body",
                "rationale": "fixed",
                "expected_bottleneck": "mixed",
                "source_template": "auto out = torch::matmul(W, X);\nreturn out + torch::matmul(A, torch::matmul(B.transpose(0, 1).contiguous(), X)); // {{BLOCK_X}}",
                "parameters": [{"name": "BLOCK_X", "values": [16, 24, 32], "default": 32}],
            },
        ]
    )

    def fake_json_prompt(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr("sysforge.workflows.optimize_lora.prompting.json_prompt", fake_json_prompt)
    family = prompting.generate_candidate_family(
        env_summary="gpu=RTX3090",
        baseline_source="baseline",
        tried_family_names=[],
        recent_body_history="(none)",
        recent_plan_summary="(none)",
        recent_body_values=[],
    )
    assert family.family_name == "good_body"


def test_revise_candidate_family_returns_new_family(monkeypatch):
    def fake_json_prompt(*args, **kwargs):
        return {
            "family_name": "family_v2",
            "rationale": "Reduce launch count.",
            "expected_bottleneck": "launch_bound",
            "source_template": "auto out = torch::matmul(W, X);\nreturn out;",
            "parameters": [],
        }

    monkeypatch.setattr("sysforge.workflows.optimize_lora.prompting.json_prompt", fake_json_prompt)
    family = prompting.revise_candidate_family(
        env_summary="gpu=RTX3090",
        family=CandidateFamilyDraft(
            family_name="family_v1",
            source_template="old",
            rationale="old",
        ),
        incumbent_source="best source",
        round_feedback="candidate=a outcome=tier1_ok",
        tried_family_names=["family_v1"],
        recent_body_history="(none)",
        recent_plan_summary="(none)",
        recent_body_values=[],
    )
    assert family.family_name == "family_v2"


def test_generate_candidate_family_retries_when_body_repeats_recent_history(monkeypatch):
    repeated_body = "auto out = torch::matmul(W, X); return out;"
    responses = iter(
        [
            {
                "family_name": "repeat_family",
                "rationale": "repeats a known body",
                "expected_bottleneck": "launch_bound",
                "source_template": "{{FORWARD_BODY}}",
                "parameters": [{"name": "FORWARD_BODY", "values": [repeated_body], "default": repeated_body}],
            },
            {
                "family_name": "fresh_family",
                "rationale": "tries a different body",
                "expected_bottleneck": "launch_bound",
                "source_template": "{{FORWARD_BODY}}",
                "parameters": [
                    {
                        "name": "FORWARD_BODY",
                        "values": ["auto dense = torch::matmul(W, X); return dense + A.sum();"],
                        "default": "auto dense = torch::matmul(W, X); return dense + A.sum();",
                    }
                ],
            },
        ]
    )

    def fake_json_prompt(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr("sysforge.workflows.optimize_lora.prompting.json_prompt", fake_json_prompt)
    family = prompting.generate_candidate_family(
        env_summary="gpu=RTX3090",
        baseline_source="baseline",
        tried_family_names=[],
        recent_body_history="(none)",
        recent_plan_summary="(none)",
        recent_body_values=[repeated_body],
        min_distinct_variants=1,
    )
    assert family.family_name == "fresh_family"


def test_generate_candidate_family_retries_when_concrete_variant_repeats_recent_history(monkeypatch):
    repeated_body = "auto out = torch::matmul(W, X); return out;"
    responses = iter(
        [
            {
                "family_name": "repeat_variant_family",
                "rationale": "repeats known body",
                "expected_bottleneck": "launch_bound",
                "source_template": "{{FORWARD_BODY}}",
                "concrete_variants": [
                    {"parameter_values": {"FORWARD_BODY": repeated_body}},
                    {"parameter_values": {"FORWARD_BODY": "auto out = torch::mm(W, X); return out;"}},
                ],
            },
            {
                "family_name": "fresh_variant_family",
                "rationale": "uses new bodies",
                "expected_bottleneck": "launch_bound",
                "source_template": "{{FORWARD_BODY}}",
                "concrete_variants": [
                    {"parameter_values": {"FORWARD_BODY": "auto dense = torch::matmul(W, X); return dense + A.sum();"}},
                    {"parameter_values": {"FORWARD_BODY": "auto dense = torch::mm(W, X); return dense + A.sum();"}},
                ],
            },
        ]
    )

    def fake_json_prompt(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr("sysforge.workflows.optimize_lora.prompting.json_prompt", fake_json_prompt)
    family = prompting.generate_candidate_family(
        env_summary="gpu=RTX3090",
        baseline_source="baseline",
        tried_family_names=[],
        recent_body_history="(none)",
        recent_plan_summary="(none)",
        recent_body_values=[repeated_body],
        min_distinct_variants=2,
    )
    assert family.family_name == "fresh_variant_family"


def test_generate_candidate_family_retries_when_body_uses_invalid_addmm_arity(monkeypatch):
    responses = iter(
        [
            {
                "family_name": "bad_addmm_family",
                "rationale": "uses invalid addmm",
                "expected_bottleneck": "launch_bound",
                "source_template": "{{FORWARD_BODY}}",
                "parameters": [
                    {
                        "name": "FORWARD_BODY",
                        "values": [
                            "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::matmul(bt, X); auto out = torch::empty({W.size(0), X.size(1)}, W.options()); torch::addmm(out, out, W, X, 1.0, 1.0); return out;"
                        ],
                        "default": "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::matmul(bt, X); auto out = torch::empty({W.size(0), X.size(1)}, W.options()); torch::addmm(out, out, W, X, 1.0, 1.0); return out;",
                    }
                ],
            },
            {
                "family_name": "good_addmm_family",
                "rationale": "uses valid addmm",
                "expected_bottleneck": "launch_bound",
                "source_template": "{{FORWARD_BODY}}",
                "parameters": [
                    {
                        "name": "FORWARD_BODY",
                        "values": [
                            "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::matmul(bt, X); auto out = torch::matmul(W, X); return torch::addmm(out, A, tmp);"
                        ],
                        "default": "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::matmul(bt, X); auto out = torch::matmul(W, X); return torch::addmm(out, A, tmp);",
                    }
                ],
            },
        ]
    )

    def fake_json_prompt(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr("sysforge.workflows.optimize_lora.prompting.json_prompt", fake_json_prompt)
    family = prompting.generate_candidate_family(
        env_summary="gpu=RTX3090",
        baseline_source="baseline",
        tried_family_names=[],
        recent_body_history="(none)",
        recent_plan_summary="(none)",
        recent_body_values=[],
        min_distinct_variants=1,
    )
    assert family.family_name == "good_addmm_family"


def test_repair_candidate_family_requires_nonempty_source(monkeypatch):
    def fake_json_prompt(*args, **kwargs):
        return {
            "family_name": "broken_fix",
            "rationale": "fix compile error",
            "expected_bottleneck": "mixed",
            "source_template": "",
            "parameters": [],
        }

    monkeypatch.setattr("sysforge.workflows.optimize_lora.prompting.json_prompt", fake_json_prompt)
    try:
        prompting.repair_candidate_family(
            env_summary="gpu=RTX3090",
            family=CandidateFamilyDraft(family_name="broken", source_template="old"),
            candidate_source="bad source",
            parameter_values={"BLOCK_X": 16},
            failure_stage="compile",
            failure_summary="missing semicolon",
        )
    except ValueError as exc:
        assert "source_template is empty" in str(exc)
    else:
        raise AssertionError("expected repair_candidate_family to reject an empty source template")
