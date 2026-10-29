"""Install the pipeline in a supported worker Python, independent of the notebook kernel."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def worker_version(python, env):
    version = json.loads(
        subprocess.check_output(
            [str(python), "-c", "import json,sys; print(json.dumps(list(sys.version_info[:2])))"],
            text=True,
            env=env,
        )
    )
    if version not in ([3, 11], [3, 12]):
        raise RuntimeError(f"Worker Python must be 3.11 or 3.12; got {version}")
    return version


def bootstrap(root, environment, extras, python_version="3.11"):
    if python_version not in ("3.11", "3.12"):
        raise ValueError("Bicleaner dependencies require worker Python 3.11 or 3.12")
    root, environment = Path(root).resolve(), Path(environment).resolve()
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    env = os.environ.copy()
    # Do not inject Colab kernel site-packages into a different Python interpreter.
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PYTHON"):
        env.pop(key, None)
    env["UV_PROJECT_ENVIRONMENT"] = str(environment)
    uv = [sys.executable, "-m", "uv"]
    try:
        if importlib.util.find_spec("uv") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True, env=env)
        command = [*uv, "sync", "--project", str(root), "--python", python_version, "--locked"]
        for extra in extras.split(","):
            command.extend(["--extra", extra])
        subprocess.run(command, check=True, env=env)
    except (OSError, subprocess.CalledProcessError):
        # Fall back to pip ONLY in a compatible worker, never in the kernel Python.
        if not python.is_file():
            candidates = [shutil.which(f"python{v}") for v in (python_version, "3.12", "3.11")]
            compatible = next((p for p in candidates if p), None)
            if compatible is None:
                raise RuntimeError(
                    "Could not provision worker Python. Check uv download/install output; "
                    "kernel Python 3.13 cannot install Bicleaner's NumPy<2 dependencies."
                ) from None
            worker_version(compatible, env)
            subprocess.run([compatible, "-m", "venv", str(environment)], check=True, env=env)
        worker_version(python, env)
        subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], check=True, env=env)
        subprocess.run(
            [str(python), "-m", "pip", "install", "-e", f"{root}[{extras}]"], check=True, env=env
        )
    version = worker_version(python, env)
    result = {"python": str(python), "version": version, "environment": str(environment)}
    manifest = environment.parent / "colab_environment.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=ROOT / ".work/colab-venv")
    parser.add_argument(
        "--extras", choices=("lid,scoring", "lid,cuda", "lid"), default="lid,scoring"
    )
    parser.add_argument("--python-version", choices=("3.11", "3.12"), default="3.11")
    args = parser.parse_args()
    result = bootstrap(ROOT, args.environment, args.extras, args.python_version)
    print("Pipeline worker:", result["python"], "version:", result["version"])


if __name__ == "__main__":
    main()
