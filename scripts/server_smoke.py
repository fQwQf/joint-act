"""Run an actual HTTP service from an exported bundle and verify a request."""

import argparse
import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    observation_path = Path(args.observation)
    with open(observation_path, encoding="utf-8") as stream:
        observation = json.load(stream)
    payload = observation.copy()
    payload["images"] = [base64.b64encode((observation_path.parent / name).read_bytes()).decode("ascii")
                         for name in observation["images"]]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with open(output / "server.log", "w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "jointact.cli", "serve", "--checkpoint", args.checkpoint,
                                    "--device", "cpu", "--port", str(port)], env=environment, stdout=log, stderr=log)
        try:
            ready = False
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("Server exited before readiness; inspect server.log")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                        health = json.load(response)
                    ready = True
                    break
                except OSError:
                    time.sleep(0.2)
            if not ready:
                raise TimeoutError("Server readiness timed out")
            request = urllib.request.Request(f"http://127.0.0.1:{port}/predict", data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=30) as response:
                predictions = json.load(response)
            result = predictions["predictions"][0]
            assert len(result["actions"]) == health["horizon"] and len(result["actions"][0]) == 7
            report = dict(passed=True, transport="actual_loopback_http", device="cpu", horizon=health["horizon"],
                          backbone=health["backbone"], fixture_only=True)
            with open(output / "report.json", "w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
            print(json.dumps(report))
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
