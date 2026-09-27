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
    tree = _main_tree()

    handler_registrations = {
        (node.args[0].id, node.args[1].id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_exception_handler"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "app"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and isinstance(node.args[1], ast.Name)
    }
    assert {
        ("HTTPException", "http_exception_handler"),
        ("RequestValidationError", "validation_exception_handler"),
        ("DetectionError", "detection_exception_handler"),
        ("Exception", "unhandled_exception_handler"),
        ("ConnectionFailure", "database_unavailable_handler"),
        ("ExecutionTimeout", "database_unavailable_handler"),
    } <= handler_registrations

    correlation_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "add_correlation_id"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "app"
    ]

    assert len(correlation_calls) == 1


def test_projects_router_is_not_registered_on_a_discarded_app():
    source = MAIN_PATH.read_text(encoding="utf-8")
    assert source.count("app.include_router(projects.router)") == 1
