"""Prepare/read Drive YAML in the worker and return JSON to any notebook kernel."""

import argparse
import json
from pathlib import Path

import yaml
from colab_ops import canonical, path_layout, validate_layout, validate_mode, validate_run

ROOT = Path(__file__).resolve().parents[1]


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


def prepare_mode_config(defaults, mode, layout, config_path=None):
    validate_mode(mode)
    validate_layout(layout)
    expected = layout[mode]
    config_path = Path(config_path) if config_path is not None else expected["config"]
    if canonical(config_path) != canonical(expected["config"]):
        raise ValueError(f"{mode}: config path must be {expected['config']}")
    if config_path.exists():
        c = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        # Existing production and smoke settings remain byte-for-byte unchanged.
        return validate_run(mode, config_path, c, layout)
    c = yaml.safe_load(Path(defaults).read_text(encoding="utf-8"))
    c["paths"].update({key: str(expected[key]) for key in ("input", "work_dir", "scratch_dir")})
    c["language_id"]["model_path"] = str(expected["work_dir"] / "models/lid.176.ftz")
    if mode == "smoke":
        c["target_size"] = 5000
        c["bicleaner"].update(candidate_size=10000, shard_size=5000)
    else:
        c["target_size"] = 300000
        c["bicleaner"].update(
            model="bitextor/bicleaner-ai-full-en-ja",
            candidate_size=1000000,
            batch_size="auto",
            device="cuda",
            shard_size=100000,
            benchmark_pairs=100000,
        )
    validate_run(mode, config_path, c, layout)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("x", encoding="utf-8") as f:
        yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False)
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defaults", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive"))
    parser.add_argument("--local-root", type=Path, default=Path("/content"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    layout = path_layout(args.repository, args.drive_root, args.local_root)
    c = prepare_mode_config(args.defaults, args.mode, layout, args.config)
    print(
        json.dumps(
            {"config_path": str(layout[args.mode]["config"]), "config": c}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
