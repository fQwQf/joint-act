"""Download the public modified-LIBERO dataset to an explicitly selected disk."""

import argparse
import json
from pathlib import Path

from jointact.utils import check_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-free-gb", type=float, default=20)
    parser.add_argument("--revision", help="Override the dataset commit in upstream.lock.json")
    parser.add_argument("--suite", choices=["all", "spatial", "object", "goal", "long"], default="all")
    args = parser.parse_args()
    check_disk(args.output, args.min_free_gb)
    from huggingface_hub import snapshot_download
    with open(Path(__file__).resolve().parents[1] / "upstream.lock.json", encoding="utf-8") as stream:
        source = json.load(stream)["modified_libero_rlds"]
    suites = dict(spatial="libero_spatial_no_noops", object="libero_object_no_noops",
                  goal="libero_goal_no_noops", long="libero_10_no_noops")
    patterns = [f"{name}/**" for key, name in suites.items() if args.suite in {"all", key}]
    Path(args.output).mkdir(parents=True, exist_ok=True)
    snapshot_download(source["id"], repo_type="dataset", revision=args.revision or source["revision"],
                      local_dir=args.output, allow_patterns=patterns, max_workers=2)


if __name__ == "__main__":
    main()
