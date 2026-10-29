"""Small argparse CLI with explicit stages and selective force."""

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

from .config import load_config, validate
from .pipeline import STAGES, Pipeline


def download_lid(c):
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    p = Path(c["language_id"]["model_path"])
    if p.exists():
        logging.info("Language ID model already exists: %s", p)
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(".partial")
    urllib.request.urlretrieve(url, temp)
    temp.replace(p)
    logging.info("Downloaded official fastText language ID model: %s", p)


def parser():
    p = argparse.ArgumentParser(description="Resumable JParaCrawl filtering for Direct S2ST")
    p.add_argument("command", choices=("run", *STAGES, "download-lid", "validate-config"))
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument(
        "--resume", action="store_true", help="Reuse complete stages and verified partial shards"
    )
    p.add_argument(
        "--force",
        action="append",
        choices=STAGES,
        default=[],
        metavar="STAGE",
        help="Rebuild this stage; repeat for several stages; descendants become stale",
    )
    p.add_argument("--until", choices=STAGES, help="Stop after this stage (run only)")
    p.add_argument("--target-size", type=int)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"))
    p.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    try:
        c = load_config(args.config)
        if args.target_size is not None:
            c["target_size"] = args.target_size
        if args.device:
            c["bicleaner"]["device"] = args.device
        validate(c)
        if args.until and args.command != "run":
            raise ValueError("--until is supported only with run")
        if args.command == "download-lid":
            download_lid(c)
        elif args.command == "validate-config":
            logging.info("Configuration valid: %s", args.config)
        else:
            Pipeline(c, resume=args.resume, force=args.force).execute(args.command, args.until)
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        logging.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
