"""Atomic trial records, restart validation, and matched-protocol comparisons."""

from contextlib import contextmanager
import itertools
import json
from pathlib import Path

from jointact.metrics import wilson_interval
from jointact.utils import atomic_json, json_digest


@contextmanager
def evaluation_lock(output):
    import fcntl
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / ".evaluation.lock", "a", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another evaluator is writing this output directory") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def expected_trials(config):
    return set(itertools.product(config["task_ids"], range(config["trials_per_task"])))


def trial_seed(seed, task_id, trial):
    return (seed + 100003 * task_id + trial) % (2**31 - 1)


def validate_record(row, config):
    key = (row["task_id"], row["trial"])
    if key not in expected_trials(config):
        raise ValueError(f"Unexpected trial identity {key}")
    if row.get("protocol_sha256") != json_digest(config):
        raise ValueError(f"Trial {key} belongs to different weights or protocol")
    if type(row.get("success")) is not bool or not 1 <= row["steps"] <= config["max_steps"]:
        raise ValueError(f"Invalid outcome for trial {key}")
    if row.get("trial_seed") != trial_seed(config["seed"], *key):
        raise ValueError(f"Invalid seed for trial {key}")
    json_digest(row)  # Reject nonfinite measurements.
    return key


def read_records(output, config):
    rows = {}
    for path in sorted((Path(output) / "trials").glob("*.json")):
        with open(path, encoding="utf-8") as stream:
            row = json.load(stream)
        key = validate_record(row, config)
        if key in rows:
            raise ValueError(f"Duplicate trial {key}")
        rows[key] = row
    return rows


def summarize(rows, config):
    records = list(rows.values())
    expected = len(expected_trials(config))
    successes = sum(row["success"] for row in records)
    per_task = {}
    for task_id in config["task_ids"]:
        subset = [row for row in records if row["task_id"] == task_id]
        count, won = len(subset), sum(row["success"] for row in subset)
        per_task[str(task_id)] = dict(successes=won, trials=count,
            success_rate=won / count if count else None, wilson95=wilson_interval(won, count) if count else None)
    return dict(successes=successes, trials=len(records), expected_trials=expected,
        success_rate=successes / len(records) if records else None,
        wilson95=wilson_interval(successes, len(records)) if records else None,
        interval_scope="binomial trial proportion; does not estimate variation across training seeds",
        per_task=per_task, completed=len(records) == expected,
        protocol_sha256=json_digest(config), policy_identity=config["policy_identity"],
        observation_orientation="180-degree rotation")


class TrialLedger:
    """Use inside evaluation_lock. Each completed trial is the source of truth."""
    def __init__(self, output, config, resume=False):
        self.output, self.config = Path(output), config
        path = self.output / "config.json"
        if resume:
            if not path.is_file():
                raise FileNotFoundError("No evaluation config to resume")
            with open(path, encoding="utf-8") as stream:
                if json.load(stream) != config:
                    raise ValueError("Cannot resume with different weights, protocol, code, or environment")
        else:
            if path.exists() or (self.output / "episodes.jsonl").exists() or (self.output / "trials").exists():
                raise FileExistsError("Choose a fresh evaluation directory or pass --resume")
            atomic_json(path, config)
        self.rows = read_records(self.output, config)
        self.publish()

    def commit(self, row):
        key = validate_record(row, self.config)
        if key in self.rows:
            raise ValueError(f"Trial {key} is already recorded")
        atomic_json(self.output / "trials" / f"task-{key[0]:03d}-trial-{key[1]:03d}.json", row)
        self.rows[key] = row
        self.publish()

    def publish(self):
        # JSONL and summary are reconstructible indexes, never the resume authority.
        temporary = self.output / ".episodes.jsonl.tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            for key in sorted(self.rows):
                stream.write(json.dumps(self.rows[key], allow_nan=False) + "\n")
        temporary.replace(self.output / "episodes.jsonl")
        result = summarize(self.rows, self.config)
        atomic_json(self.output / "summary.json", result)
        return result


def compare_libero(runs, output):
    """Compare complete policies on exactly the same observations/protocol/initial states."""
    if len(runs) < 2:
        raise ValueError("Need at least two labelled evaluations")
    checked, reference = {}, None
    controls = ("protocol_version", "suite", "trials_per_task", "task_ids", "seed", "max_steps", "settle_steps",
                "execute_horizon", "observation", "initial_state_assets", "task_assets", "precision")
    for label, directory in runs.items():
        directory = Path(directory)
        with open(directory / "config.json", encoding="utf-8") as stream:
            config = json.load(stream)
        if config.get("protocol_version") != "jointact-libero-v2":
            raise ValueError(f"{label}: unsupported evaluation protocol")
        rows = read_records(directory, config)
        result = summarize(rows, config)
        if not result["completed"]:
            raise ValueError(f"{label}: evaluation is incomplete ({len(rows)}/{result['expected_trials']})")
        matched = {key: config[key] for key in controls}
        if reference is None:
            reference = matched
        elif reference != matched:
            changed = [key for key in controls if reference[key] != matched[key]]
            raise ValueError(f"{label}: unmatched evaluation controls: {changed}")
        checked[label] = dict(config=config, rows=rows, summary=result)
    pairs = []
    for a, b in itertools.combinations(checked, 2):
        left, right = checked[a]["rows"], checked[b]["rows"]
        a_only = sum(left[key]["success"] and not right[key]["success"] for key in left)
        b_only = sum(right[key]["success"] and not left[key]["success"] for key in left)
        both = sum(left[key]["success"] and right[key]["success"] for key in left)
        pairs.append(dict(a=a, b=b, trials=len(left), a_only_successes=a_only, b_only_successes=b_only,
                          both_successes=both, neither_successes=len(left)-a_only-b_only-both,
                          paired_success_delta_a_minus_b=(a_only-b_only) / len(left)))
    result = dict(protocol=reference, runs={label: dict(summary=row["summary"],
        training_data_sha256=row["config"]["training_data_sha256"], policy_config=row["config"]["policy_config"],
        source_sha256=row["config"]["source_sha256"]) for label, row in checked.items()}, pairs=pairs,
        scope="paired evaluation outcomes; inspect recorded training configurations before attributing causality")
    atomic_json(output, result)
    return result
