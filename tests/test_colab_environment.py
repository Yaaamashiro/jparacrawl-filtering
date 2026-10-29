import ast
import importlib.util
import json
import subprocess
from pathlib import Path

import nbformat
import pytest
from conftest import ROOT


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_targets_supported_worker_not_kernel(tmp_path, monkeypatch):
    setup = script("bootstrap_colab")
    calls = []
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(setup.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    monkeypatch.setattr(setup.subprocess, "check_output", lambda *a, **kw: "[3, 11]")
    monkeypatch.setenv("PYTHONPATH", "/kernel313/site-packages")
    target = tmp_path / "worker"
    result = setup.bootstrap(ROOT, target, "lid,scoring")
    command, options = calls[0]
    assert command[2:4] == ["uv", "sync"]
    assert command[command.index("--python") + 1] == "3.11"
    assert "--system" not in command
    assert options["env"]["UV_PROJECT_ENVIRONMENT"] == str(target)
    assert "PYTHONPATH" not in options["env"]
    assert Path(result["python"]).is_relative_to(target)
    assert json.loads((tmp_path / "colab_environment.json").read_text())["version"] == [3, 11]


def test_pip_fallback_also_uses_worker(tmp_path, monkeypatch):
    setup = script("bootstrap_colab")
    target = tmp_path / "worker"
    python = target / ("Scripts/python.exe" if setup.os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    calls = []
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(setup.subprocess, "check_output", lambda *a, **kw: "[3, 11]")

    def run(command, **kw):
        calls.append(command)
        if "sync" in command:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(setup.subprocess, "run", run)
    setup.bootstrap(ROOT, target, "lid,scoring")
    pip_install = next(c for c in calls if c[1:4] == ["-m", "pip", "install"])
    assert pip_install[0] == str(python)
    assert pip_install[-1] == f"{ROOT}[lid,scoring]"


def test_bootstrap_rejects_unsupported_worker(tmp_path):
    setup = script("bootstrap_colab")
    with pytest.raises(ValueError, match="3.11 or 3.12"):
        setup.bootstrap(ROOT, tmp_path / "worker", "lid", "3.13")


def test_colab_config_preserves_existing_drive_settings(tmp_path):
    helper = script("prepare_colab_config")
    config = tmp_path / "config.yaml"
    defaults = ROOT / "configs/default.yaml"
    first = helper.prepare_config(defaults, config, tmp_path / "input.tsv", tmp_path)
    assert first["paths"]["input"] == str(tmp_path / "input.tsv")
    config.write_text(
        config.read_text(encoding="utf-8").replace("target_size: 300000", "target_size: 123"),
        encoding="utf-8",
    )
    preserved = helper.prepare_config(defaults, config, tmp_path / "other.tsv", tmp_path)
    assert preserved["target_size"] == 123
    assert preserved["paths"]["input"] == first["paths"]["input"]


def notebook():
    return nbformat.read(ROOT / "notebooks/jparacrawl_filtering_colab.ipynb", as_version=4)


def test_notebook_stage_commands_run_in_worker(tmp_path):
    cell = next(
        c for c in notebook().cells if c.cell_type == "code" and "def run_stage" in c.source
    )
    function = next(
        n
        for n in ast.parse(cell.source).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_stage"
    )
    calls = []

    class Runner:
        @staticmethod
        def run(command, **kwargs):
            calls.append((command, kwargs))

    namespace = dict(
        subprocess=Runner,
        PIPELINE_PYTHON="/worker311/bin/python",
        CONFIG=tmp_path / "config.yaml",
        WORKER_ENV={"PATH": "/worker311/bin"},
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<notebook>", "exec"), namespace)
    namespace["run_stage"]("score", "--resume")
    command, options = calls[0]
    assert command[:4] == ["/worker311/bin/python", "-m", "jparacrawl_filtering", "score"]
    assert options["env"] == namespace["WORKER_ENV"]


def test_notebook_kernel_does_not_import_worker_dependencies():
    for cell in notebook().cells:
        if cell.cell_type != "code":
            continue
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, ast.ImportFrom):
                assert not node.module.startswith(
                    ("jparacrawl_filtering", "yaml", "fasttext", "tensorflow")
                )
            elif isinstance(node, ast.Import):
                assert all(
                    n.name not in ("yaml", "fasttext", "tensorflow", "pyarrow") for n in node.names
                )
