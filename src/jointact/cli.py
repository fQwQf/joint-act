"""Command-line entry points for the complete data/model/evaluation lifecycle."""

import argparse
import json
from pathlib import Path


def _runtime_arguments(parser):
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--allow-shared-gpu", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(prog="jointact")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Inspect GPU occupation, storage and optional packages")
    doctor.add_argument("--path", default=".")
    doctor.add_argument("--output")
    fixture = sub.add_parser("fixture", help="Generate engineering-only synthetic episodes")
    fixture.add_argument("--output", required=True)
    fixture.add_argument("--episodes", type=int, default=12)
    fixture.add_argument("--length", type=int, default=16)
    fixture.add_argument("--image-size", type=int, default=32)
    fixture.add_argument("--cameras", type=int, default=2)
    fixture.add_argument("--seed", type=int, default=42)
    for name in ("convert-hdf5", "convert-rlds"):
        converter = sub.add_parser(name)
        converter.add_argument("--source", required=True)
        converter.add_argument("--output", required=True)
        converter.add_argument("--seed", type=int, default=42)
        converter.add_argument("--val-fraction", type=float, default=0.1)
        converter.add_argument("--test-fraction", type=float, default=0.1)
        converter.add_argument("--max-episodes", type=int)
        converter.add_argument("--min-free-gb", type=float, default=2.0)
        converter.add_argument("--camera-view", choices=["both", "primary"], default="both")
        if name == "convert-hdf5":
            converter.add_argument("--rotate-images", action="store_true")
        else:
            converter.add_argument("--primary-key", default="image")
            converter.add_argument("--wrist-key", default="wrist_image")
            converter.add_argument("--state-key", default="state")
    audit = sub.add_parser("validate-data")
    audit.add_argument("--root", required=True)
    audit.add_argument("--skip-hashes", action="store_true")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--horizon", type=int, default=8)
    prepare.add_argument("--num-modes", type=int, default=64)
    prepare.add_argument("--max-samples", type=int, default=100000)
    prepare.add_argument("--iterations", type=int, default=40)
    prepare.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--resume")
    train.add_argument("--device")
    train.add_argument("--stop-after", type=int, help="Checkpoint and stop early without changing the LR schedule")
    study = sub.add_parser("prepare-study", help="Generate matched joint/regression/ablation configurations")
    study.add_argument("--config", required=True)
    study.add_argument("--output", required=True)
    study.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    study.add_argument("--variants", nargs="+", choices=["joint", "regression", "prototype", "no_brier", "aligned", "cost", "aligned_cost"],
                       default=["joint", "regression", "prototype", "no_brier"])
    study.add_argument("--joint-parent", help="Explicit objective fork inheriting optimizer/sampler/RNG and step")
    study.add_argument("--regression-parent", help="Matched regression checkpoint for a continuation study")
    run = sub.add_parser("run-study", help="Train, resume, evaluate offline and export each configured policy")
    run.add_argument("--plan", required=True)
    run.add_argument("--device")
    run.add_argument("--world-size", type=int, default=1)
    run.add_argument("--max-runs", type=int)
    compare = sub.add_parser("compare-libero", help="Compare paired trials after checking evaluation controls")
    compare.add_argument("--run", action="append", required=True, help="LABEL=EVALUATION_DIRECTORY; repeat per policy")
    compare.add_argument("--output", required=True)
    offline = sub.add_parser("evaluate-offline")
    _runtime_arguments(offline)
    offline.add_argument("--root")
    offline.add_argument("--artifacts")
    offline.add_argument("--split", choices=["val", "test"], default="test")
    offline.add_argument("--batch-size", type=int, default=8)
    offline.add_argument("--max-batches", type=int)
    offline.add_argument("--output", required=True)
    for name in ("predict", "benchmark"):
        prediction = sub.add_parser(name)
        _runtime_arguments(prediction)
        prediction.add_argument("--observation", required=True, help="JSON with images (paths), proprio, instruction")
        prediction.add_argument("--output")
        if name == "benchmark":
            prediction.add_argument("--warmup", type=int, default=5)
            prediction.add_argument("--repeats", type=int, default=30)
            prediction.add_argument("--execute-horizon", type=int)
    benchmark = sub.add_parser("benchmark-dataset", help="Measure varied held-out observations at batch size one")
    _runtime_arguments(benchmark)
    benchmark.add_argument("--root")
    benchmark.add_argument("--split", choices=["val", "test"], default="test")
    benchmark.add_argument("--observations", type=int, default=8)
    benchmark.add_argument("--warmup", type=int, default=5)
    benchmark.add_argument("--repeats", type=int, default=30)
    benchmark.add_argument("--execute-horizon", type=int)
    benchmark.add_argument("--output", required=True)
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    serve = sub.add_parser("serve")
    _runtime_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--max-batch", type=int, default=16)
    libero = sub.add_parser("evaluate-libero")
    _runtime_arguments(libero)
    libero.add_argument("--suite", default="libero_spatial")
    libero.add_argument("--output", required=True)
    libero.add_argument("--trials", type=int, default=50)
    libero.add_argument("--seed", type=int, default=42)
    libero.add_argument("--task-ids", type=int, nargs="+")
    libero.add_argument("--execute-horizon", type=int)
    libero.add_argument("--max-steps", type=int)
    libero.add_argument("--video", action="store_true")
    libero.add_argument("--min-free-gb", type=float, default=2)
    libero.add_argument("--resume", action="store_true")
    libero.add_argument("--max-new-trials", type=int, help="Stop after this many additional trials; resume keeps the full protocol")
    calibration = sub.add_parser("calibrate")
    calibration.add_argument("--checkpoint", required=True)
    calibration.add_argument("--root")
    calibration.add_argument("--artifacts")
    calibration.add_argument("--device", default="cpu")
    calibration.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    calibration.add_argument("--allow-shared-gpu", action="store_true")
    calibration.add_argument("--max-samples", type=int, default=10000)
    calibration.add_argument("--batch-size", type=int, default=16)
    teacher = sub.add_parser("score-teacher")
    teacher.add_argument("--root", required=True)
    teacher.add_argument("--artifacts", required=True)
    teacher.add_argument("--output", required=True)
    teacher.add_argument("--model-id", required=True)
    teacher.add_argument("--revision")
    teacher.add_argument("--unnorm-key")
    teacher.add_argument("--device", default="cpu")
    teacher.add_argument("--temperature", type=float, default=1)
    teacher.add_argument("--max-samples", type=int)
    teacher.add_argument("--allow-shared-gpu", action="store_true")
    teacher.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    return parser


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    command = args.pop("command")
    if command == "doctor":
        import importlib.util
        import platform
        import shutil
        from jointact.utils import atomic_json, gpu_inventory
        usage = shutil.disk_usage(args["path"])
        try:
            gpus = gpu_inventory()
        except (OSError, RuntimeError) as error:
            gpus = dict(unavailable=str(error))
        result = dict(python=platform.python_version(), path=str(Path(args["path"]).resolve()),
                      free_gib=usage.free / 1024**3, gpus=gpus,
                      optional_packages={name: importlib.util.find_spec(name) is not None
                                         for name in ("transformers", "peft", "tensorflow", "libero", "fastapi")})
        if args["output"]:
            atomic_json(args["output"], result)
    elif command == "fixture":
        from jointact.fixture import create_fixture
        result = create_fixture(**args)
    elif command.startswith("convert-"):
        from jointact.data.convert import convert_hdf5, convert_rlds
        result = (convert_hdf5 if command == "convert-hdf5" else convert_rlds)(**args)
    elif command == "validate-data":
        from jointact.data.prepare import validate_dataset
        result = validate_dataset(args["root"], not args["skip_hashes"])
    elif command == "prepare":
        from jointact.data.prepare import prepare_artifacts
        result = prepare_artifacts(**args)
    elif command == "train":
        from jointact.config import Config
        from jointact.training import train
        config = Config.load(args["config"])
        if args["resume"]:
            config.train.resume = args["resume"]
        if args["device"]:
            config.train.device = args["device"]
        result = train(config, stop_after=args["stop_after"])
    elif command == "prepare-study":
        from jointact.study import prepare_study
        result = prepare_study(**args)
    elif command == "run-study":
        from jointact.study import run_study
        result = run_study(**args)
    elif command == "compare-libero":
        from jointact.evaluation.records import compare_libero
        runs = {}
        for value in args["run"]:
            label, separator, directory = value.partition("=")
            if not separator or not label or not directory or label in runs:
                raise ValueError("Each --run must have a unique LABEL=EVALUATION_DIRECTORY")
            runs[label] = directory
        result = compare_libero(runs, args["output"])
    elif command == "export":
        from jointact.checkpoint import export_bundle
        result = dict(bundle=export_bundle(**args))
    elif command == "score-teacher":
        from jointact.teacher import score_teacher
        result = score_teacher(**args)
    elif command == "calibrate":
        from jointact.calibration import calibrate
        result = calibrate(**args)
    else:
        from jointact.inference import PolicyRuntime
        from jointact.utils import atomic_json
        runtime = PolicyRuntime(**{key: args.pop(key) for key in ("checkpoint", "device", "precision", "allow_shared_gpu")})
        if command in {"predict", "benchmark"}:
            path = Path(args.pop("observation"))
            with open(path, encoding="utf-8") as stream:
                observation = json.load(stream)
            observation["images"] = [str((path.parent / name).resolve()) for name in observation["images"]]
            output = args.pop("output")
            result = runtime.predict(observation) if command == "predict" else runtime.benchmark(observation, **args)
            if output:
                atomic_json(output, result)
        elif command == "benchmark-dataset":
            from jointact.inference import benchmark_dataset
            result = benchmark_dataset(runtime, **args)
        elif command == "evaluate-offline":
            from torch.utils.data import DataLoader
            from jointact.data.dataset import EpisodeDataset
            from jointact.training import evaluate
            dataset = EpisodeDataset(args["root"] or runtime.config.data.root, runtime.config.model.horizon,
                                     args["split"], args["artifacts"] or runtime.config.data.artifacts)
            loader = DataLoader(dataset, batch_size=args["batch_size"], collate_fn=runtime.collator)
            result = dict(split=args["split"], probability_semantics="behavior_mode_distribution",
                          policy_identity=runtime.identity, model_config=runtime.config.model.__dict__,
                          manifest_sha256=runtime.metadata["manifest_sha256"],
                          **evaluate(runtime.model, loader, runtime.device, runtime.precision, args["max_batches"]))
            dataset.close()
            atomic_json(args["output"], result)
        elif command == "serve":
            import uvicorn
            from jointact.serve import create_app
            uvicorn.run(create_app(runtime, max_batch=args.pop("max_batch")), **args)
            return
        elif command == "evaluate-libero":
            from jointact.evaluation.libero import evaluate_libero
            result = evaluate_libero(runtime, **args)
        else:
            raise ValueError(f"Unknown command: {command}")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
