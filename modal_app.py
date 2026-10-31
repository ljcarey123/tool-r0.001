"""Run Tool-R0 training and eval on a Modal cloud GPU.

    pip install modal && modal token new                      # one-time setup
    modal run modal_app.py::diagnose                          # ~1-2 GPU min: is the Generator healthy?
    modal run --detach modal_app.py::train --preset dev       # train (keeps running if you disconnect)
    modal run modal_app.py::evaluate --run dev                 # trained vs base, synthetic eval
    modal volume get tool-r0-runs dev ./runs                   # download adapters + logs

Each run writes to its own folder (default: the preset name) on the `tool-r0-runs` volume.
Training is resume-safe: re-running the same command continues from the last iteration.
"""
import subprocess

import modal

# Tried in order when capacity is short. L4/A10G use bf16; T4 has no bf16 (falls back to fp16)
GPU = ["L4", "A10G", "T4"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1",
        "transformers==5.9.0",
        "trl==1.5.1",
        "peft==0.21.2",
        "accelerate==1.13.0",
        "datasets==4.8.5",
        "pydantic==2.13.4",
    )
    .env({"HF_HOME": "/cache/hf", "PYTHONPATH": "/root/src", "PYTHONUTF8": "1"})
    .add_local_dir("src", "/root/src")
    .add_local_dir("scripts", "/root/scripts")
)

app = modal.App("tool-r0", image=image)
runs = modal.Volume.from_name("tool-r0-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("tool-r0-hf-cache", create_if_missing=True)
VOLUMES: dict = {"/runs": runs, "/cache": hf_cache}


def _cli(*args: str) -> None:
    subprocess.run(["python", "-m", "tool_r0.main", *args], check=True)


@app.function(gpu=GPU, volumes=VOLUMES, timeout=24 * 3600)
def train(preset: str = "dev", run: str = "", extra: str = "") -> None:
    """`extra` passes further CLI flags, e.g. --extra "--generator-steps 20"."""
    ckpt = f"/runs/{run or preset}"
    try:
        _cli("train", "--preset", preset, "--checkpoint-dir", ckpt, *extra.split())
    finally:
        runs.commit()
        hf_cache.commit()


@app.function(gpu=GPU, volumes=VOLUMES, timeout=1800)
def diagnose(temps: str = "0.7,0.85,1.0", n: int = 32, run: str = "") -> None:
    """Generator parse/validity rates per temperature (~1-2 GPU minutes). `run` uses trained adapters."""
    args = ["--temps", temps, "--n", str(n)] + (["--checkpoint", f"/runs/{run}"] if run else [])
    subprocess.run(["python", "/root/scripts/diag_generator.py", *args], check=True)


@app.function(gpu=GPU, volumes=VOLUMES, timeout=4 * 3600)
def evaluate(run: str = "dev", benchmark: str = "synthetic", n: int = 100) -> None:
    _cli("eval", "--checkpoint", f"/runs/{run}/solver", "--benchmark", benchmark, "--n", str(n))
