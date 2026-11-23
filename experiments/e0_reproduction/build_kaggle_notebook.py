"""Build the self-contained Kaggle notebook for E0 from the code committed at HEAD.

The notebook clones the public upstream repository and unpacks the E0 code over it, so it needs no
access token for this repository.

Usage: python build_kaggle_notebook.py [--output kaggle_e0.ipynb]
"""

import argparse
import base64
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUNDLED_FILES = [
    "experiments/e0_reproduction/check_walker.py",
    "experiments/e0_reproduction/reproduce_zero_shot.py",
    "experiments/paper_analysis/paper_tables.json",
    "experiments/requirements-gpu.txt",
]

INSTRUCTIONS = """# E0: reproduce FLOCK zero-shot results

1. In the notebook settings, select the **GPU T4** accelerator and turn **Internet on**.
2. Choose **Save Version**, then **Save & Run All**.
3. Download the `e0_*` files from the Output tab.

The notebook checks the walker, runs three small inductive graphs for comparison with Table 9, and
times 20 Hetionet triples at the paper's zero-shot setting (n = 512, P = 2)."""

SETUP = """import base64, io, json, os, subprocess, tarfile
from pathlib import Path

EXPERIMENT_COMMIT = "__COMMIT__"
UPSTREAM_COMMIT = "__UPSTREAM__"
WORK = Path(os.environ.get("E0_WORK", "/tmp/e0"))
OUTPUT = Path(os.environ.get("E0_OUTPUT", "/kaggle/working"))
REPO = WORK / "flock"
PYTHON = Path(os.environ.get("E0_PYTHON", WORK / "env/bin/python"))
SCRIPTS = REPO / "experiments/e0_reproduction"
RUN_ENV = {
    **os.environ,
    "EXPERIMENT_COMMIT": EXPERIMENT_COMMIT,
    "CUDA_VISIBLE_DEVICES": "0",
    "TQDM_DISABLE": "1",
}
WORK.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)


def sh(command, check=True, cwd=None):
    print("$", command, flush=True)
    bash_command = f"set -o pipefail; {command}"
    code = subprocess.run(bash_command, shell=True, executable="/bin/bash", cwd=cwd, env=RUN_ENV).returncode
    if code and check:
        raise RuntimeError(f"command failed with exit code {code}")
    if code:
        print(f"FAILED with exit code {code}; continuing", flush=True)
    return code == 0"""

GPU_CHECK = """gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True
).stdout
print(gpu)
assert "T4" in gpu, "Select the GPU T4 accelerator in the notebook settings."
sh("nproc")"""

GET_CODE = """if not REPO.exists():
    sh(f"git clone -q https://github.com/jw9730/flock {REPO}")
sh(f"git -C {REPO} checkout -q {UPSTREAM_COMMIT}")
BUNDLE = "__BUNDLE__"
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz") as archive:
    archive.extractall(REPO)
print("code ready at", REPO)"""

INSTALL = """sh("pip install -q uv")
sh(f"uv venv -q --python 3.12 {WORK / 'env'}")
sh(f"uv pip install -q --python {PYTHON} --index-url https://download.pytorch.org/whl/cu126 torch==2.8.0")
sh(f"uv pip install -q --python {PYTHON} -r {REPO}/experiments/requirements-gpu.txt")
sh(
    f"uv pip install -q --python {PYTHON} --only-binary :all: torch-scatter==2.1.2 "
    "-f https://data.pyg.org/whl/torch-2.8.0+cu126.html"
)
sh(f"uv pip install -q --python {PYTHON} --no-deps .", cwd=REPO / "graph-walker")
sh(f"{PYTHON} -c 'import torch, graph_walker; print(torch.__version__, torch.cuda.get_device_name(0))'", cwd=WORK)
sh(f"uv pip freeze --python {PYTHON} > {OUTPUT}/e0_gpu_freeze.txt")"""

WALKER = """sh(
    f"{PYTHON} check_walker.py --output {OUTPUT}/e0_walker_check_kaggle.json "
    f"2>&1 | tee {OUTPUT}/e0_walker_check_kaggle.log",
    check=False,
    cwd=SCRIPTS,
)"""

REPRODUCE = """sh(
    f"{PYTHON} reproduce_zero_shot.py --device cuda --batch-size 8 --data-root {WORK}/data "
    "--datasets FB15k237Inductive:v1,WN18RRInductive:v1,NELLInductive:v1 "
    f"--output {OUTPUT}/e0_zero_shot.json 2>&1 | tee {OUTPUT}/e0_zero_shot.log",
    check=False,
    cwd=SCRIPTS,
)"""

HETIONET = """sh(
    f"{PYTHON} reproduce_zero_shot.py --device cuda --batch-size 1 --max-triples 20 --data-root {WORK}/data "
    f"--datasets Hetionet --output {OUTPUT}/e0_hetionet_timing.json 2>&1 | tee {OUTPUT}/e0_hetionet_timing.log",
    check=False,
    cwd=SCRIPTS,
)"""

SUMMARY = """for name in ("e0_zero_shot.json", "e0_hetionet_timing.json"):
    path = OUTPUT / name
    if not path.exists():
        print(name, "is missing; check its log")
        continue
    for run in json.loads(path.read_text())["runs"]:
        print(
            run["dataset"], "| MRR", round(run["measured"]["mrr"], 3), "| paper", run["paper_all_relations"]["mrr"],
            "| full run", run["full_run"], "| s/triple", round(run["seconds_per_triple"], 2),
            "| peak GPU GB", round(run["peak_gpu_memory_gb"], 2),
        )"""


def git(*arguments, text=True):
    return subprocess.run(["git", *arguments], cwd=REPO, capture_output=True, text=text, check=True).stdout


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("kaggle_e0.ipynb"))
    args = parser.parse_args()

    upstream = git("rev-parse", "upstream/main").strip()
    if git("diff", "--name-only", upstream, "HEAD", "--", "src_entity", "graph-walker", "checkpoints").strip():
        raise SystemExit("upstream code differs from HEAD, so a clean upstream clone would not match this commit")

    archive = git("archive", "--format=tar.gz", "HEAD", "--", *BUNDLED_FILES, text=False)
    get_code = GET_CODE.replace("__BUNDLE__", base64.b64encode(archive).decode())
    setup = SETUP.replace("__COMMIT__", git("rev-parse", "HEAD").strip()).replace("__UPSTREAM__", upstream)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": INSTRUCTIONS.splitlines(keepends=True)},
            *map(code_cell, (setup, GPU_CHECK, get_code, INSTALL, WALKER, REPRODUCE, HETIONET, SUMMARY)),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    args.output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"wrote {args.output} from {git('rev-parse', '--short', 'HEAD').strip()}")


if __name__ == "__main__":
    main()
