"""Initialize an explicit LIBERO configuration without its interactive import."""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="LIBERO repository checkout")
    parser.add_argument("--config-dir", required=True, help="Use this as LIBERO_CONFIG_PATH")
    parser.add_argument("--datasets", required=True)
    args = parser.parse_args()
    benchmark = (Path(args.source) / "libero" / "libero").resolve()
    for name in ("bddl_files", "init_files", "assets"):
        if not (benchmark / name).is_dir():
            raise FileNotFoundError(benchmark / name)
    destination = Path(args.config_dir)
    if (destination / "config.yaml").exists():
        raise FileExistsError("Existing LIBERO configuration is preserved")
    destination.mkdir(parents=True, exist_ok=True)
    Path(args.datasets).mkdir(parents=True, exist_ok=True)
    values = dict(benchmark_root=str(benchmark), bddl_files=str(benchmark / "bddl_files"),
                  init_states=str(benchmark / "init_files"), assets=str(benchmark / "assets"),
                  datasets=str(Path(args.datasets).resolve()))
    with open(destination / "config.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump(values, stream)
    print(f"LIBERO_CONFIG_PATH={destination.resolve()}")


if __name__ == "__main__":
    main()
