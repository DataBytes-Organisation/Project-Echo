"""Application wiring regression tests for TD002 (C16.1)."""

import ast
from pathlib import Path


MAIN_PATH = (
    Path(__file__).resolve().parents[3] / "production" / "backend" / "app" / "main.py"
)


def _main_tree():
    return ast.parse(MAIN_PATH.read_text(encoding="utf-8"))


def test_main_constructs_exactly_one_fastapi_application():
    constructors = [
        node
        for node in ast.walk(_main_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FastAPI"
    ]
    assert len(constructors) == 1


def test_error_and_correlation_wiring_target_the_served_app():
    calls = {
        node.func.id
        for node in ast.walk(_main_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "app"
    }
    assert "register_exception_handlers" in calls
    assert "add_correlation_id" in calls


def test_projects_router_is_not_registered_on_a_discarded_app():
    source = MAIN_PATH.read_text(encoding="utf-8")
    assert source.count("app.include_router(projects.router)") == 1
