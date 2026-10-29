"""Pin a Hugging Face model snapshot on Drive for reproducible offline inference."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="bitextor/bicleaner-ai-full-en-ja")
    parser.add_argument(
        "--revision", default="main", help="Prefer a full commit SHA for later reruns"
    )
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(args.model, revision=args.revision)
    dest = args.destination / info.sha
    snapshot_download(repo_id=args.model, revision=info.sha, local_dir=dest)
    (dest / "source_revision.json").write_text(
        json.dumps({"model": args.model, "revision": info.sha}, indent=2), encoding="utf-8"
    )
    print(dest.resolve())


if __name__ == "__main__":
    main()
