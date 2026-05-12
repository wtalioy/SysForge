import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent / "sysforge"


def test_no_workflow_orchestration_imports_integrations_llm():
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "sysforge.integrations.llm":
                raise AssertionError(str(path))
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "sysforge.integrations.llm":
                        raise AssertionError(str(path))


def test_only_agent_modules_import_shared_llm_helpers():
    for path in (ROOT / "workflows").rglob("*.py"):
        if path.name == "agent.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "sysforge.agent.llm":
                if any(alias.name.startswith("chat_") for alias in node.names):
                    raise AssertionError(str(path))
            if isinstance(node, ast.ImportFrom) and node.module == "sysforge.agent.prompts":
                if any(alias.name.endswith("_prompt") for alias in node.names):
                    raise AssertionError(str(path))
