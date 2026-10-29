"""Prepare/read Drive YAML in the worker and return JSON to any notebook kernel."""

import argparse
import json
from pathlib import Path

import yaml


def prepare_config(defaults, config, input_path, work):
    work.mkdir(parents=True, exist_ok=True)
    if not config.exists():
        c = yaml.safe_load(defaults.read_text(encoding="utf-8"))
        c["paths"].update(
            input=str(input_path),
            work_dir=str(work),
            scratch_dir="/content/jparacrawl-filtering-scratch",
        )
        c["language_id"]["model_path"] = str(work / "models/lid.176.ftz")
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yaml.safe_load(config.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defaults", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_config(args.defaults, args.config, args.input, args.work_dir),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
