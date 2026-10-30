"""Colab orchestration only; standard library, safe in a Python 3.13 kernel."""

import gzip
import itertools
import json
import random
import shlex
import subprocess
import tempfile
from pathlib import Path


def validate_mode(mode):
    if mode not in ("smoke", "production"):
        raise ValueError("RUN_MODE must be 'smoke' or 'production'")
    return mode


def path_layout(repository, drive_root=Path("/content/drive/MyDrive"), local_root=Path("/content")):
    repository, drive_root, local_root = map(Path, (repository, drive_root, local_root))
    work = drive_root / "jparacrawl-filtering"
    return {
        "production": {
            "config": work / "config.yaml",
            "input": drive_root / "jparacrawl/v3/en-ja/en-ja.bicleaner05.txt",
            "work_dir": work,
            "scratch_dir": local_root / "jparacrawl-filtering-scratch",
        },
        "smoke": {
            "config": repository / "configs/smoke.yaml",
            "input": drive_root / "jparacrawl/smoke/en-ja-smoke.txt",
            "work_dir": drive_root / "jparacrawl-filtering-smoke",
            "scratch_dir": local_root / "jparacrawl-filtering-smoke-scratch",
        },
    }


def canonical(path):
    return Path(path).expanduser().resolve()


def validate_layout(layout):
    smoke, production = layout["smoke"], layout["production"]
    for key in ("config", "input", "work_dir", "scratch_dir"):
        if canonical(smoke[key]) == canonical(production[key]):
            raise ValueError(f"Smoke and production {key} must be separate")
    # Resolve symlinks and reject overlapping output trees, not just equal names.
    for key in ("work_dir", "scratch_dir"):
        a, b = canonical(smoke[key]), canonical(production[key])
        if a.is_relative_to(b) or b.is_relative_to(a):
            raise ValueError(f"Smoke and production {key} must not overlap")
    for key in ("config", "input", "work_dir", "scratch_dir"):
        if canonical(smoke[key]).is_relative_to(canonical(production["work_dir"])):
            raise ValueError(f"Smoke {key} must not be inside production work_dir")


def validate_run(mode, config_path, config, layout):
    validate_mode(mode)
    validate_layout(layout)
    expected = layout[mode]
    actual = {**config["paths"], "config": config_path}
    for key in ("config", "input", "work_dir", "scratch_dir"):
        if key not in actual or canonical(actual[key]) != canonical(expected[key]):
            raise ValueError(f"{mode}: {key} must be {expected[key]}; got {actual.get(key)}")
    return config


def nvidia_gpu(env=None):
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def gpu_extras(gpu):
    return "lid,cuda" if gpu else "lid,scoring"


def stream_command(command, env):
    command = [str(x) for x in command]
    # Colab's inline backend belongs to the kernel, not the isolated worker.
    # Set this before matplotlib is imported; matplotlib.use() would be too late.
    env = {**env, "MPLBACKEND": "Agg"}
    print("Running:\n" + shlex.join(command), flush=True)
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
        shell=False,
    )
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
        returncode = proc.wait()
    except BaseException:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    finally:
        proc.stdout.close()
    if returncode != 0:
        raise RuntimeError(f"Command failed with return code {returncode}: {shlex.join(command)}")
    return returncode


def pipeline_command(python, stage, config, options):
    # Overrides would make the printed config differ from the actual CLI config.
    for option in options:
        if any(
            str(option) == flag or str(option).startswith(flag + "=")
            for flag in ("--config", "--device", "--target-size")
        ):
            raise ValueError("Edit the selected config for config/device/target-size changes")
    return [str(python), "-m", "jparacrawl_filtering", stage, "--config", str(config), *options]


def worker_json(python, program, env):
    return json.loads(
        subprocess.check_output(
            [str(python), "-c", program],
            text=True,
            env=env,
            timeout=30,
        )
    )


def preflight(mode, config_path, config, layout, python, env, repository):
    validate_run(mode, config_path, config, layout)
    version = worker_json(
        python, "import json,platform; print(json.dumps(platform.python_version()))", env
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
        timeout=15,
    ).strip()
    values = {
        "mode": mode,
        "config": str(config_path),
        **{k: config["paths"][k] for k in ("input", "work_dir", "scratch_dir")},
        "target_size": config["target_size"],
        "candidate_size": config["bicleaner"]["candidate_size"],
        "device": config["bicleaner"]["device"],
        "worker_python": version,
        "worker_executable": str(python),
        "git_commit": commit,
    }
    print("=== JParaCrawl Pipeline Preflight ===", flush=True)
    for key, value in values.items():
        print(f"{key:<17}: {value}", flush=True)
    print("====================================", flush=True)
    return values


def gpu_preflight(python, config, env):
    gpu = nvidia_gpu(env)
    info = worker_json(
        python,
        """
import json, shutil
from importlib import metadata
from jparacrawl_filtering.scoring import cuda_available
try:
    version = metadata.version('bicleaner-ai')
except metadata.PackageNotFoundError:
    version = None
print(json.dumps({'cuda_available': cuda_available(),
                  'cli': shutil.which('bicleaner-ai-classify'), 'version': version}))
""",
        env,
    )
    print("GPU:\n" + (gpu or "Not detected"), flush=True)
    print(f"cuda_available = {info['cuda_available']}", flush=True)
    print(f"Bicleaner:\nCLI = {info['cli']}\nversion = {info['version']}", flush=True)
    if not info["cli"] or not info["version"]:
        raise RuntimeError("Bicleaner CLI/package missing in worker; rerun worker bootstrap")
    if config["bicleaner"]["device"] == "cuda" and (not gpu or not info["cuda_available"]):
        raise RuntimeError(
            "device=cuda requires nvidia-smi and worker cuda_available(); scoring stopped"
        )
    return info


def will_score(stage, options):
    if stage == "score":
        return True
    if stage != "run":
        return False
    options = list(options)
    until = None
    for i, option in enumerate(options):
        if option == "--until" and i + 1 < len(options):
            until = options[i + 1]
        elif str(option).startswith("--until="):
            until = str(option).split("=", 1)[1]
    return until not in ("normalize", "rule-filter", "speech-filter", "exact-dedup")


def create_smoke_sample(
    source, destination, layout, sample_mode="head", size=50000, seed=42, overwrite=False
):
    validate_layout(layout)
    if canonical(source) != canonical(layout["production"]["input"]):
        raise ValueError("Smoke sample source must be the configured production input")
    if canonical(destination) != canonical(layout["smoke"]["input"]):
        raise ValueError("Smoke sample destination must be the separate smoke input")
    if sample_mode not in ("head", "random"):
        raise ValueError("SMOKE_SAMPLE_MODE must be 'head' or 'random'")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("Smoke sample size must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Smoke sample seed must be a nonnegative integer")
    source, destination = Path(source), Path(destination)
    manifest = destination.with_suffix(destination.suffix + ".sample.json")
    if destination.exists() and not overwrite and not manifest.exists():
        print(
            "Existing smoke input preserved; sampling provenance unavailable. No sampling performed.",
            flush=True,
        )
        return None
    stat = source.stat()
    settings = {
        "source": str(canonical(source)),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "sample_mode": sample_mode,
        "size": size,
        "seed": seed,
    }
    if destination.exists() and not overwrite:
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous["settings"] != settings:
            raise ValueError(
                "Existing smoke sample has different settings; set REBUILD_SMOKE_INPUT=True explicitly"
            )
        print(f"Reusing smoke sample: {destination} ({previous['sampled']} rows)", flush=True)
        return previous["sampled"]
    print(
        f"Smoke sampling: {sample_mode}, size={size}, seed={seed}\nSource: {source}\nDestination: {destination}",
        flush=True,
    )
    if sample_mode == "random":
        print("Random sampling scans the entire input with a bounded reservoir.", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if source.suffix == ".gz" else open
    temporary = None
    try:
        with (
            opener(source, "rb") as inp,
            tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=destination.name + ".",
                suffix=".partial",
                delete=False,
            ) as out,
        ):
            temporary = Path(out.name)
            if sample_mode == "head":
                sampled = 0
                for line in itertools.islice(inp, size):
                    out.write(line)
                    sampled += 1
            else:
                rng, reservoir = random.Random(seed), []
                for index, line in enumerate(inp):
                    if len(reservoir) < size:
                        reservoir.append((index, line))
                    else:
                        slot = rng.randrange(index + 1)
                        if slot < size:
                            reservoir[slot] = (index, line)
                    if (index + 1) % 1000000 == 0:
                        print(f"Smoke sampling scanned={index + 1:,}", flush=True)
                for _, line in sorted(reservoir):
                    out.write(line)
                sampled = len(reservoir)
        temporary.replace(destination)
        manifest.write_text(
            json.dumps({"settings": settings, "sampled": sampled}, indent=2), encoding="utf-8"
        )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Smoke input ready: {sampled:,} rows", flush=True)
    return sampled
