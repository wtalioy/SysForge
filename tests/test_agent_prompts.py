from pathlib import Path

from sysforge.agent.prompts import render_prompt


def test_render_prompt_preserves_escaped_braces_and_substitutes_variables():
    prompt_dir = Path(__file__).resolve().parent.parent / "sysforge" / "workflows" / "profiling" / "prompts"

    rendered = render_prompt(
        prompt_dir,
        "extract_value.txt",
        target="dram_bw",
        unit="GB/s",
        parse_hint="look for RESULT",
        stdout="RESULT dram_bw=12.3 unit=GB/s",
    )

    assert "target `dram_bw`" in rendered
    assert '"value": <number or null>' in rendered
    assert '{{' not in rendered
    assert '}}' not in rendered
