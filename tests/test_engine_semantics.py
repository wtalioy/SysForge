import os

import pytest
import torch

from sysforge.workflows.optimize_runtime.correctness import (
    build_stress_events,
    compare_engine_with_reference,
    run_correctness_case,
)
from sysforge.workflows.optimize_runtime.models import EngineStrategy
from sysforge.workflows.optimize_runtime.renderer import write_engine
from sysforge.workflows.optimize_runtime.runtime_io import load_engine_module, resolve_device


def normal(shape, scale=0.02):
    return torch.randn(*shape, dtype=torch.float32) * scale


def write_toy_weights(config, weight_dir):
    torch.manual_seed(1234)
    vocab_size = int(config["vocab_size"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_layers = int(config["num_hidden_layers"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    state = {"embed_tokens.weight": normal((vocab_size, hidden_size))}
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}"
        state[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden_size)
        state[f"{prefix}.self_attn.q_proj.weight"] = normal((num_heads * head_dim, hidden_size))
        state[f"{prefix}.self_attn.k_proj.weight"] = normal((num_kv_heads * head_dim, hidden_size))
        state[f"{prefix}.self_attn.v_proj.weight"] = normal((num_kv_heads * head_dim, hidden_size))
        state[f"{prefix}.self_attn.o_proj.weight"] = normal((hidden_size, num_heads * head_dim))
        state[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(hidden_size)
        state[f"{prefix}.mlp.gate_proj.weight"] = normal((intermediate_size, hidden_size))
        state[f"{prefix}.mlp.up_proj.weight"] = normal((intermediate_size, hidden_size))
        state[f"{prefix}.mlp.down_proj.weight"] = normal((hidden_size, intermediate_size))
    state["norm.weight"] = torch.ones(hidden_size)
    state["lm_head.weight"] = normal((vocab_size, hidden_size))
    weight_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, weight_dir / "model.pt")


def tiny_config(*, kv_heads=2):
    return {
        "model_type": "tiny_llama",
        "torch_dtype": "float16",
        "vocab_size": 97,
        "hidden_size": 24,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": kv_heads,
        "head_dim": 6,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
    }


@pytest.mark.skipif(bool(os.environ.get("ENGINE_UNDER_TEST")), reason="external engine path uses public config only")
def test_rendered_kv_cache_engine_matches_reference_on_gqa(tmp_path):
    device = resolve_device(os.environ.get("ENGINE_SEMANTICS_DEVICE", "auto"))
    config = tiny_config(kv_heads=2)
    weight_dir = tmp_path / "weights"
    write_toy_weights(config, weight_dir)
    engine_path = tmp_path / "engine.py"
    write_engine(engine_path, EngineStrategy(prefill_policy="group_by_length"))

    compare_engine_with_reference(
        engine_path=str(engine_path),
        model_config=config,
        weight_dir=str(weight_dir),
        events=build_stress_events(config["vocab_size"], device),
        device=device,
    )


@pytest.mark.skipif(bool(os.environ.get("ENGINE_UNDER_TEST")), reason="external engine path uses public config only")
def test_rendered_recompute_engine_matches_reference(tmp_path):
    device = resolve_device(os.environ.get("ENGINE_SEMANTICS_DEVICE", "auto"))
    config = tiny_config(kv_heads=1)
    weight_dir = tmp_path / "weights"
    write_toy_weights(config, weight_dir)
    engine_path = tmp_path / "engine.py"
    write_engine(
        engine_path,
        EngineStrategy(prefill_policy="per_request", kv_policy="none"),
    )

    compare_engine_with_reference(
        engine_path=str(engine_path),
        model_config=config,
        weight_dir=str(weight_dir),
        events=build_stress_events(config["vocab_size"], device),
        device=device,
    )


@pytest.mark.skipif(bool(os.environ.get("ENGINE_UNDER_TEST")), reason="external engine path uses public config only")
def test_rendered_pad_batch_engine_matches_reference_on_mixed_lengths(tmp_path):
    device = resolve_device(os.environ.get("ENGINE_SEMANTICS_DEVICE", "auto"))
    config = tiny_config(kv_heads=2)
    weight_dir = tmp_path / "weights"
    write_toy_weights(config, weight_dir)
    engine_path = tmp_path / "engine.py"
    write_engine(
        engine_path,
        EngineStrategy(
            prefill_policy="pad_batch",
            kv_policy="per_request_prealloc",
            attention_policy="manual",
        ),
    )

    compare_engine_with_reference(
        engine_path=str(engine_path),
        model_config=config,
        weight_dir=str(weight_dir),
        events=build_stress_events(config["vocab_size"], device),
        device=device,
    )


@pytest.mark.skipif(bool(os.environ.get("ENGINE_UNDER_TEST")), reason="external engine path uses public config only")
def test_rendered_advanced_search_strategy_matches_reference(tmp_path):
    device = resolve_device(os.environ.get("ENGINE_SEMANTICS_DEVICE", "auto"))
    config = tiny_config(kv_heads=2)
    weight_dir = tmp_path / "weights"
    write_toy_weights(config, weight_dir)
    engine_path = tmp_path / "engine.py"
    write_engine(
        engine_path,
        EngineStrategy(
            prefill_policy="group_by_length",
            kv_policy="per_request_prealloc",
            attention_policy="manual",
            decode_attention_policy="sdpa_by_length",
            cache_growth_policy="decode_slack_128",
            cache_layout_policy="transposed_k",
            norm_policy="triton_rmsnorm",
        ),
    )

    compare_engine_with_reference(
        engine_path=str(engine_path),
        model_config=config,
        weight_dir=str(weight_dir),
        events=build_stress_events(config["vocab_size"], device),
        device=device,
    )


def test_rendered_engine_rejects_duplicate_request_ids(tmp_path):
    device = resolve_device(os.environ.get("ENGINE_SEMANTICS_DEVICE", "auto"))
    config = tiny_config(kv_heads=1)
    weight_dir = tmp_path / "weights"
    write_toy_weights(config, weight_dir)
    engine_path = tmp_path / "engine.py"
    write_engine(engine_path, EngineStrategy(prefill_policy="group_by_length"))
    engine = load_engine_module(str(engine_path)).create_engine(config, str(weight_dir), device)

    with pytest.raises(ValueError, match="unique"):
        engine.prefill(
            [1, 1],
            [
                torch.randint(0, config["vocab_size"], (4,), device=device),
                torch.randint(0, config["vocab_size"], (4,), device=device),
            ],
        )

    engine.prefill([1, 2], [torch.randint(0, config["vocab_size"], (4,), device=device) for _ in range(2)])
    with pytest.raises(ValueError, match="unique"):
        engine.decode([1, 1], torch.randint(0, config["vocab_size"], (2,), device=device))


def test_external_engine_public_stress():
    engine_path = os.environ.get("ENGINE_UNDER_TEST")
    if not engine_path:
        pytest.skip("ENGINE_UNDER_TEST is not set")
    result = run_correctness_case(
        engine_path=engine_path,
        model_config_path="target/model_config.json",
        weight_dir="target/weights",
        case="stress",
        device=os.environ.get("ENGINE_SEMANTICS_DEVICE", "auto"),
    )
    assert result["status"] == "passed"
