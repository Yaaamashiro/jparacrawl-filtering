import ast
import json
import os
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import colab_ops as ops
import nbformat
import pytest
import yaml
from conftest import ROOT, demo
from prepare_colab_config import prepare_mode_config

from jparacrawl_filtering.pipeline import Pipeline


@pytest.fixture
def layout(tmp_path):
    return ops.path_layout(tmp_path / "repo", tmp_path / "drive", tmp_path / "local")


def make_config(mode, layout):
    return prepare_mode_config(ROOT / "configs/default.yaml", mode, layout)


@pytest.mark.parametrize("mode", ["smoke", "production"])
def test_mode_matches_paths(mode, layout):
    c = make_config(mode, layout)
    assert ops.validate_run(mode, layout[mode]["config"], c, layout) is c


@pytest.mark.parametrize(
    "mode,key,other",
    [
        ("smoke", "input", "production"),
        ("smoke", "work_dir", "production"),
        ("smoke", "scratch_dir", "production"),
        ("production", "work_dir", "smoke"),
        ("production", "input", "smoke"),
        ("production", "scratch_dir", "smoke"),
    ],
)
def test_mode_path_mismatch_fails(mode, key, other, layout):
    c = make_config(mode, layout)
    c["paths"][key] = str(layout[other][key])
    with pytest.raises(ValueError, match=key):
        ops.validate_run(mode, layout[mode]["config"], c, layout)


@pytest.mark.parametrize("mode", ["test", "SMOKE", "", None])
def test_invalid_mode_creates_nothing(mode, layout):
    with pytest.raises(ValueError, match="RUN_MODE"):
        prepare_mode_config(ROOT / "configs/default.yaml", mode, layout)
    assert not layout["production"]["config"].exists()
    assert not layout["smoke"]["config"].exists()


def test_config_paths_and_sizes_are_separate(layout):
    production = make_config("production", layout)
    before = layout["production"]["config"].read_bytes()
    smoke = make_config("smoke", layout)
    for key in ("input", "work_dir", "scratch_dir"):
        assert smoke["paths"][key] != production["paths"][key]
    assert (
        smoke["target_size"],
        smoke["bicleaner"]["candidate_size"],
        smoke["bicleaner"]["shard_size"],
    ) == (5000, 10000, 5000)
    assert (
        production["target_size"],
        production["bicleaner"]["candidate_size"],
        production["bicleaner"]["shard_size"],
    ) == (300000, 1000000, 100000)
    assert production["bicleaner"]["device"] == "cuda"
    assert layout["production"]["config"].read_bytes() == before


def test_existing_config_preserved_and_wrong_config_rejected(layout):
    c = make_config("production", layout)
    c["target_size"] = 1234
    p = layout["production"]["config"]
    p.write_text(yaml.safe_dump(c), encoding="utf-8")
    before = p.read_bytes()
    assert make_config("production", layout)["target_size"] == 1234
    assert p.read_bytes() == before
    with pytest.raises(ValueError, match="config path"):
        prepare_mode_config(ROOT / "configs/default.yaml", "smoke", layout, p)
    assert p.read_bytes() == before


def test_nested_work_trees_rejected(layout):
    bad = deepcopy(layout)
    bad["smoke"]["work_dir"] = layout["production"]["work_dir"] / "smoke"
    with pytest.raises(ValueError, match="overlap"):
        ops.validate_layout(bad)


@pytest.mark.parametrize("available,expected", [(True, "lid,cuda"), (False, "lid,scoring")])
def test_gpu_extra_selection(available, expected, monkeypatch):
    monkeypatch.setattr(
        ops.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0 if available else 1,
            stdout="NVIDIA L4, 23034 MiB" if available else "",
        ),
    )
    gpu = ops.nvidia_gpu()
    assert ops.gpu_extras(gpu) == expected


def test_missing_nvidia_binary_selects_cpu(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(ops.subprocess, "run", missing)
    assert ops.gpu_extras(ops.nvidia_gpu()) == "lid,scoring"


@pytest.mark.parametrize("returncode", [0, 7])
def test_stream_runner_is_incremental(returncode, monkeypatch, capsys):
    observed = []
    env = {"PATH": "/worker311/bin", "PYTHONUNBUFFERED": "1"}

    class Lines:
        closed = False

        def __iter__(self):
            yield "stage=scoring processed=5000 total=21923 pairs/sec=133.00\n"
            # Reader must display the first line before asking for the next one.
            observed.append(capsys.readouterr().out)
            assert "processed=5000" in observed[-1]
            yield "stage=scoring processed=10000 estimated_remaining_seconds=89.6\n"

        def close(self):
            self.closed = True

    lines = Lines()

    def popen(command, **kwargs):
        assert command[0] == "/worker311/bin/python"
        assert kwargs["env"] is env and kwargs["shell"] is False
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["stdout"] == subprocess.PIPE and kwargs["bufsize"] == 1
        return SimpleNamespace(stdout=lines, wait=lambda: returncode)

    monkeypatch.setattr(ops.subprocess, "Popen", popen)
    command = ops.pipeline_command(
        "/worker311/bin/python", "score", "/smoke/config.yaml", ["--resume"]
    )
    if returncode:
        with pytest.raises(RuntimeError, match="return code 7"):
            ops.stream_command(command, env)
    else:
        assert ops.stream_command(command, env) == 0
    assert "Running:" in observed[0] and "--resume" in observed[0]
    assert "processed=10000" in capsys.readouterr().out
    assert lines.closed


def test_real_runner_merges_stdout_and_stderr(capsys):
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    program = "import sys; print('stage=scoring processed=15000 total=21923 pairs/sec=137.52', flush=True); print('estimated_remaining_seconds=50.3', file=sys.stderr, flush=True)"
    ops.stream_command([sys.executable, "-c", program], env)
    output = capsys.readouterr().out
    assert "processed=15000" in output and "estimated_remaining_seconds=50.3" in output


@pytest.mark.parametrize(
    "option", ["--config", "--config=/production/config.yaml", "--device=cuda", "--target-size"]
)
def test_hidden_cli_config_overrides_rejected(option):
    with pytest.raises(ValueError, match="selected config"):
        ops.pipeline_command("worker", "score", "smoke.yaml", [option])


def test_preflight_prints_actual_config(mode="smoke", *, layout, monkeypatch, capsys):
    c = make_config(mode, layout)
    monkeypatch.setattr(ops, "worker_json", lambda *a: "3.11.18")
    monkeypatch.setattr(ops.subprocess, "check_output", lambda *a, **kw: "abcdef123\n")
    values = ops.preflight(mode, layout[mode]["config"], c, layout, "worker311", {}, ROOT)
    output = capsys.readouterr().out
    for key in (
        "mode",
        "config",
        "input",
        "work_dir",
        "scratch_dir",
        "target_size",
        "candidate_size",
        "device",
        "worker_python",
        "git_commit",
    ):
        assert key in output and str(values[key]) in output


@pytest.mark.parametrize("bad", ["nvidia", "cuda_available", "cli", "version"])
def test_cuda_gpu_preflight_failure_stops(layout, monkeypatch, capsys, bad):
    c = make_config("production", layout)
    info = {"cuda_available": True, "cli": "/worker/bin/bicleaner-ai-classify", "version": "3.6"}
    if bad != "nvidia":
        info[bad] = None
    monkeypatch.setattr(
        ops, "nvidia_gpu", lambda *a: "" if bad == "nvidia" else "NVIDIA L4, 23034 MiB"
    )
    monkeypatch.setattr(ops, "worker_json", lambda *a: info)
    with pytest.raises(RuntimeError):
        ops.gpu_preflight("worker311", c, {})
    assert "Bicleaner:" in capsys.readouterr().out


def test_auto_can_pass_gpu_preflight_without_gpu(layout, monkeypatch):
    c = make_config("smoke", layout)
    monkeypatch.setattr(ops, "nvidia_gpu", lambda *a: "")
    monkeypatch.setattr(
        ops,
        "worker_json",
        lambda *a: {"cuda_available": False, "cli": "worker-cli", "version": "3.6"},
    )
    assert ops.gpu_preflight("worker311", c, {})["cuda_available"] is False


def test_cpu_run_does_not_require_gpu_preflight():
    assert not ops.will_score("run", ["--resume", "--until", "exact-dedup"])
    assert not ops.will_score("run", ["--until=exact-dedup"])
    assert ops.will_score("run", ["--resume"])
    assert ops.will_score("score", ["--resume"])


def write_source(layout, count=200):
    source = layout["production"]["input"]
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"".join(f"synthetic row {i}\n".encode() for i in range(count)))
    return source, layout["smoke"]["input"]


def test_head_sample_and_reuse(layout):
    source, dest = write_source(layout)
    assert ops.create_smoke_sample(source, dest, layout, size=17) == 17
    assert dest.read_bytes() == b"".join(source.read_bytes().splitlines(keepends=True)[:17])
    before = dest.stat().st_mtime_ns
    assert ops.create_smoke_sample(source, dest, layout, size=17) == 17
    assert dest.stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="different settings"):
        ops.create_smoke_sample(source, dest, layout, sample_mode="random", size=17)


def test_random_sample_reproducible_and_bounded(layout):
    source, dest = write_source(layout)
    ops.create_smoke_sample(source, dest, layout, sample_mode="random", size=17, seed=42)
    first = dest.read_bytes()
    assert len(first.splitlines()) == 17
    assert first != b"".join(source.read_bytes().splitlines(keepends=True)[:17])
    ops.create_smoke_sample(
        source, dest, layout, sample_mode="random", size=17, seed=42, overwrite=True
    )
    assert dest.read_bytes() == first
    ops.create_smoke_sample(
        source, dest, layout, sample_mode="random", size=17, seed=43, overwrite=True
    )
    assert dest.read_bytes() != first


def test_manual_smoke_input_preserved(layout):
    source, dest = write_source(layout)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"existing synthetic sample\n")
    assert ops.create_smoke_sample(source, dest, layout) is None
    assert dest.read_bytes() == b"existing synthetic sample\n"
    with pytest.raises(ValueError):
        ops.create_smoke_sample(source, source, layout, overwrite=True)


def test_synthetic_smoke_cpu_does_not_write_production(layout):
    production = make_config("production", layout)
    samples = json.loads((ROOT / "tests/fixtures/synthetic.json").read_text(encoding="utf-8"))[:7]
    source, dest = write_source(layout)
    source.write_text(
        "".join(
            "\t".join(["synthetic.invalid", "synthetic.invalid", "0.5", r["en"], r["ja"]]) + "\n"
            for r in samples
        ),
        encoding="utf-8",
    )
    prod_files = {
        p.relative_to(layout["production"]["work_dir"]): p.read_bytes()
        for p in layout["production"]["work_dir"].rglob("*")
        if p.is_file()
    }
    source_before = source.read_bytes()
    ops.create_smoke_sample(source, dest, layout, size=7)
    c = make_config("smoke", layout)
    stores = Pipeline(c, lid=demo.SyntheticLanguageID()).execute(until="exact-dedup")
    assert stores["exact-dedup"].counts()["accepted"] == 6
    assert c["paths"]["work_dir"] != production["paths"]["work_dir"]
    assert prod_files == {
        p.relative_to(layout["production"]["work_dir"]): p.read_bytes()
        for p in layout["production"]["work_dir"].rglob("*")
        if p.is_file()
    }
    assert source.read_bytes() == source_before


def test_notebook_mode_and_preflight_structure():
    nb = nbformat.read(ROOT / "notebooks/jparacrawl_filtering_colab.ipynb", as_version=4)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    assert "RUN_MODE = 'smoke'" in code[0]
    with pytest.raises(ValueError, match="RUN_MODE"):
        exec(code[0].replace("RUN_MODE = 'smoke'", "RUN_MODE = 'invalid'"), {})
    bootstrap = next(c for c in code if "scripts/bootstrap_colab.py" in c)
    assert "ops.gpu_extras" in bootstrap and "USE_CUDA_DEPS" not in bootstrap
    runner = next(c for c in code if "def run_stage" in c)
    tree = ast.parse(runner)
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_stage"
    )
    calls = [
        n.func.attr
        for n in ast.walk(function)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert calls.index("validate_run") < calls.index("stream_command")
    assert "preflight" in calls and "gpu_preflight" in calls
    assert "read_active_config" in runner


def notebook_runner():
    nb = nbformat.read(ROOT / "notebooks/jparacrawl_filtering_colab.ipynb", as_version=4)
    cell = next(c for c in nb.cells if c.cell_type == "code" and "def run_stage" in c.source)
    return next(
        n
        for n in ast.parse(cell.source).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_stage"
    )


@pytest.mark.parametrize(
    "stage,options",
    [
        ("run", ("--resume", "--until", "exact-dedup")),
        ("score", ("--resume",)),
        ("select", ("--resume",)),
    ],
)
def test_notebook_preflight_precedes_each_long_stage(stage, options, layout):
    current = make_config("production", layout)
    events = []
    helpers = SimpleNamespace(
        validate_run=ops.validate_run,
        pipeline_command=ops.pipeline_command,
        will_score=ops.will_score,
        preflight=lambda *args: events.append(("preflight", args[2]["target_size"])),
        gpu_preflight=lambda *args: events.append(("gpu", None)),
        stream_command=lambda *args, **kwargs: events.append(("stream", args[0][0])),
    )
    namespace = dict(
        ops=helpers,
        read_active_config=lambda: current,
        RUN_MODE="production",
        CONFIG=layout["production"]["config"],
        LAYOUT=layout,
        PIPELINE_PYTHON="worker311",
        WORKER_ENV={},
        REPO_DIR=ROOT,
    )
    exec(
        compile(ast.Module(body=[notebook_runner()], type_ignores=[]), "<notebook>", "exec"),
        namespace,
    )
    current["target_size"] = 12345  # A config edit after notebook variable initialization.
    namespace["run_stage"](stage, *options)
    assert events[0] == ("preflight", 12345)
    assert events[-1] == ("stream", "worker311")
    assert any(name == "gpu" for name, _ in events) == (stage == "score")


def test_notebook_stale_mode_stops_before_launch(layout):
    wrong = make_config("production", layout)
    launched = []
    helpers = SimpleNamespace(
        validate_run=ops.validate_run, stream_command=lambda *a, **kw: launched.append(a)
    )
    namespace = dict(
        ops=helpers,
        read_active_config=lambda: wrong,
        RUN_MODE="smoke",
        CONFIG=layout["smoke"]["config"],
        LAYOUT=layout,
    )
    exec(
        compile(ast.Module(body=[notebook_runner()], type_ignores=[]), "<notebook>", "exec"),
        namespace,
    )
    with pytest.raises(ValueError):
        namespace["run_stage"]("score", "--resume")
    assert not launched
