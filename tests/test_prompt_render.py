from sysforge.profiling import prompting


def test_all_prompts_load():
    for name in [
        "system.txt", "generate_probe.txt", "fix_compile_error.txt",
        "fix_runtime_error.txt", "fix_implausible.txt", "extract_value.txt",
        "analyze_metrics.txt", "generate_gemm.txt",
    ]:
        assert prompting.load_prompt(name)


def test_generate_probe_renders():
    out = prompting.render_prompt(
        "generate_probe.txt",
        target="dram_latency_cycles", category="latency", unit="cycles",
        strategy="pointer_chase_dram",
        description="DRAM latency via pointer chase",
        plausible_min=200, plausible_max=1500,
        hints="{}",
        run_timeout_s=30, run_timeout_s_safe=25, num_trials=32,
    )
    assert "dram_latency_cycles" in out
    assert "pointer_chase_dram" in out


def test_all_prompts_render_with_placeholders_only():
    """Ensure no template contains stray '{...}' that str.format would trip over."""
    for name, kwargs in [
        ("system.txt", {}),
        ("generate_probe.txt", dict(target="t", category="c", unit="u", strategy="s",
                                    description="d", plausible_min=0, plausible_max=1,
                                    hints="{}", run_timeout_s=30, run_timeout_s_safe=25,
                                    num_trials=32)),
        ("fix_compile_error.txt", dict(target="t", cmd="c", stdout="", stderr="",
                                       source="", history="")),
        ("fix_runtime_error.txt", dict(target="t", rc=0, wallclock_s="0", timed_out="False",
                                       stdout="", stderr="", source="", history="")),
        ("fix_implausible.txt", dict(target="t", value="1", unit="u",
                                     plausible_min=0, plausible_max=2, reason="r",
                                     stdout="", source="", history="")),
        ("extract_value.txt", dict(target="t", unit="u", parse_hint="", stdout="")),
        ("analyze_metrics.txt", dict(metrics="[]", ncu_csv="")),
    ]:
        prompting.render_prompt(name, **kwargs)
