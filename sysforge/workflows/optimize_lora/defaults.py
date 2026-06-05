from __future__ import annotations

from dataclasses import dataclass


PUBLIC_D_MIN = 3584
PUBLIC_D_MAX = 4608
DEFAULT_RANK = 16


@dataclass(frozen=True)
class LoraHarnessConfig:
    validation_shape: int = 4096
    rank: int = DEFAULT_RANK
    seed: int = 0
    warmup: int = 2
    iters: int = 5
    enforce_public_range: bool = True
    tier1_shapes: tuple[int, ...] = (4096, 4608)
    tier2_shapes: tuple[int, ...] = (3584, 4096, 4608)
    tier3_shapes: tuple[int, ...] = (3584, 4096, 4608)
    screen_warmup: int = 1
    screen_iters: int = 2


@dataclass(frozen=True)
class SearchConfig:
    max_family_variants: int = 6
    min_seed_variants: int = 3
    max_full_evaluations_per_round: int = 3
    max_llm_rounds: int = 6
    final_confirmation_candidates: int = 3
    max_close_finalists: int = 4
    clear_winner_speedup: float = 1.05
    profile_enabled: bool = True
    final_confirm_warmup: int = 2
    final_confirm_iters: int = 6
    tier1_rerun_warmup: int = 2
    tier1_rerun_iters: int = 4
    tier1_rerun_band_pct: float = 1.5
    max_stalled_rounds: int = 3


DEFAULT_LORA_HARNESS_CONFIG = LoraHarnessConfig()
DEFAULT_LORA_SEARCH_CONFIG = SearchConfig()
