"""Build the self-contained E1 and E2 GPU notebooks from the code committed at HEAD.

Each notebook clones the public FLOCK and ULTRA repositories at pinned commits and unpacks this study's
code over them, so it needs no access token for this repository.

Usage, from experiments/: uv run python gpu/build_notebooks.py
"""

import argparse
import base64
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO / "experiments/notebooks"
ULTRA_COMMIT = "427966ad8ed60420eef034063d44f3153addff90"
BUNDLED_PATHS = ["experiments/common", "experiments/gpu", "experiments/splits", "experiments/requirements-gpu.txt"]

HOW_TO_RUN = """- **Kaggle:** select the **GPU T4** accelerator, turn **Internet on**, choose **Save Version**, then
  **Save & Run All**. Download the files from the Output tab.
- **Colab:** choose **Runtime > Change runtime type > T4 GPU**, then **Runtime > Run all**. Keep the tab
  open. The last cell downloads a zip of the results.

Each fold is saved as it finishes, and no new fold starts after `TIME_BUDGET_HOURS`."""

E1_INTRO = f"""# E1 and E6: FLOCK and ULTRA on Hetionet

Runs, in priority order:

1. A check of the public ULTRA checkpoint against its official results.
2. E6: relation queries on the 56 Compound–Disease test triples.
3. E1: drug → disease queries on the shared folds.
4. E1: disease → drug queries on the shared folds.

{HOW_TO_RUN}"""

E2_INTRO = f"""# E2: ablation of *resembles* edges

Reruns FLOCK and ULTRA on the drug → disease queries after removing all *resembles* edges from the graph.
Compare it with the E1 notebook, which uses the same folds and seeds on the full graph.

{HOW_TO_RUN}"""

E2_SEED2_INTRO = f"""# E2: third FLOCK seed of the *resembles* ablation

Adds walk seed 2 to the E2 ablation. Run it only if E1 completed seed 2 of drug → disease, because the two
are compared seed by seed. ULTRA is deterministic, so the main E2 notebook already covers it.

{HOW_TO_RUN}"""

SETUP = """import base64, io, json, os, shutil, subprocess, tarfile, time
from pathlib import Path

TIME_BUDGET_HOURS = 8
NOTEBOOK = "__NAME__"
EXPERIMENT_COMMIT = "__COMMIT__"
UPSTREAM_COMMIT = "__UPSTREAM__"
ULTRA_COMMIT = "__ULTRA__"


def running_on_colab():
    try:
        import google.colab
    except ImportError:
        return False
    return "KAGGLE_KERNEL_RUN_TYPE" not in os.environ


ON_COLAB = running_on_colab()
WORK = Path(os.environ.get("GPU_WORK", "/tmp/gpu_run"))
OUTPUT = Path(os.environ.get("GPU_OUTPUT", f"/content/{NOTEBOOK}_output" if ON_COLAB else "/kaggle/working"))
REPO = WORK / "flock"
ULTRA = WORK / "ULTRA"
PYTHON = Path(os.environ.get("GPU_PYTHON", WORK / "env/bin/python"))
DEADLINE = time.time() + TIME_BUDGET_HOURS * 3600
RUN_ENV = {
    **os.environ,
    "EXPERIMENT_COMMIT": EXPERIMENT_COMMIT,
    "CUDA_VISIBLE_DEVICES": "0",
    "TQDM_DISABLE": "1",
    "PYTHONPATH": str(REPO / "experiments"),
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
    return code == 0


def stage(module, arguments):
    sh(
        f"{PYTHON} -m {module} {arguments} --deadline {DEADLINE} 2>&1 | tee -a {OUTPUT}/{NOTEBOOK}.log",
        check=False,
        cwd=REPO / "experiments",
    )"""

GPU_CHECK = """gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True
).stdout
print(gpu)
assert "T4" in gpu, "Select the GPU T4 accelerator in the notebook settings."
sh("nproc")"""

GET_CODE = """if not REPO.exists():
    sh(f"git clone -q https://github.com/jw9730/flock {REPO}")
sh(f"git -C {REPO} checkout -q {UPSTREAM_COMMIT}")
if not ULTRA.exists():
    sh(f"git clone -q https://github.com/DeepGraphLearning/ULTRA {ULTRA}")
sh(f"git -C {ULTRA} checkout -q {ULTRA_COMMIT}")
BUNDLE = "__BUNDLE__"
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz") as archive:
    archive.extractall(REPO)
sh(f"bash {REPO}/experiments/common/get_hetionet.sh")
print("code and data ready")"""

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
sh(f"uv pip freeze --python {PYTHON} > {OUTPUT}/{NOTEBOOK}_freeze.txt")"""

E1_STAGES = [
    """stage(
    "gpu.check_ultra",
    f"--ultra-repo {ULTRA} --data-root {WORK}/ultra_data --output {OUTPUT}/ultra_checkpoint_check.json",
)""",
    """stage("gpu.run_relation", f"--model ultra --ultra-repo {ULTRA} --output-dir {OUTPUT}/runs")
stage("gpu.run_relation", f"--model flock --seeds 0,1,2 --output-dir {OUTPUT}/runs")""",
    """stage("gpu.run_entity", f"--model ultra --ultra-repo {ULTRA} --direction tail --output-dir {OUTPUT}/runs")
stage("gpu.run_entity", f"--model flock --direction tail --seeds 0,1,2 --output-dir {OUTPUT}/runs")""",
    """stage("gpu.run_entity", f"--model ultra --ultra-repo {ULTRA} --direction head --output-dir {OUTPUT}/runs")
stage("gpu.run_entity", f"--model flock --direction head --seeds 0,1,2 --output-dir {OUTPUT}/runs")""",
]

E2_STAGES = [
    """stage(
    "gpu.run_entity",
    f"--model ultra --ultra-repo {ULTRA} --direction tail --condition no_resembles --output-dir {OUTPUT}/runs",
)""",
    """stage(
    "gpu.run_entity",
    f"--model flock --direction tail --condition no_resembles --seeds 0,1 --output-dir {OUTPUT}/runs",
)""",
]

E2_SEED2_STAGES = [
    """stage(
    "gpu.run_entity",
    f"--model flock --direction tail --condition no_resembles --seeds 2 --output-dir {OUTPUT}/runs",
)""",
]

NOTEBOOKS = {
    "gpu_e1": (E1_INTRO, E1_STAGES),
    "gpu_e2": (E2_INTRO, E2_STAGES),
    "gpu_e2_seed2": (E2_SEED2_INTRO, E2_SEED2_STAGES),
}

SUMMARY = """sh(f"{PYTHON} -m gpu.summarise_runs --runs {OUTPUT}/runs 2>&1 | tee {OUTPUT}/{NOTEBOOK}_summary.txt",
   check=False, cwd=REPO / "experiments")"""

DOWNLOAD = """if ON_COLAB:
    from google.colab import files

    files.download(shutil.make_archive(f"/content/{NOTEBOOK}_results", "zip", OUTPUT))"""


def git(*arguments, text=True):
    return subprocess.run(["git", *arguments], cwd=REPO, capture_output=True, text=text, check=True).stdout


def cell(index, source, cell_type="code"):
    outputs = {"execution_count": None, "outputs": []} if cell_type == "code" else {}
    return {
        "cell_type": cell_type,
        "id": str(index),
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **outputs,
    }


def build(name, intro, stages, get_code, commit, upstream, summary=SUMMARY):
    setup = (
        SETUP.replace("__NAME__", name)
        .replace("__COMMIT__", commit)
        .replace("__UPSTREAM__", upstream)
        .replace("__ULTRA__", ULTRA_COMMIT)
    )
    sources = (setup, GPU_CHECK, get_code, INSTALL, *stages, *([summary] if summary else []), DOWNLOAD)
    notebook = {
        "cells": [
            cell(0, intro, "markdown"),
            *(cell(index, source) for index, source in enumerate(sources, start=1)),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = OUTPUT_DIR / f"{name}.ipynb"
    # Same layout and cell ids as nbstripout, so the hook leaves a freshly built notebook unchanged.
    path.write_text(json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(REPO)} from {commit[:7]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", choices=[[], *NOTEBOOKS], help="Notebooks to build (default: all)")
    args = parser.parse_args()
    upstream = git("rev-parse", "upstream/main").strip()
    if git("diff", "--name-only", upstream, "HEAD", "--", "src_entity", "src_relation", "graph-walker", "checkpoints"):
        raise SystemExit("upstream code differs from HEAD, so a clean upstream clone would not match this commit")
    archive = git("archive", "--format=tar.gz", "HEAD", "--", *BUNDLED_PATHS, text=False)
    get_code = GET_CODE.replace("__BUNDLE__", base64.b64encode(archive).decode())
    commit = git("rev-parse", "HEAD").strip()
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name in args.names or NOTEBOOKS:
        intro, stages = NOTEBOOKS[name]
        build(name, intro, stages, get_code, commit, upstream)


if __name__ == "__main__":
    main()
