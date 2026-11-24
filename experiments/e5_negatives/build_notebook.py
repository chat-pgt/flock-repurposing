"""Build the self-contained E5 GPU notebooks from the code committed at HEAD.

Usage, from experiments/: uv run python -m e5_negatives.build_notebook
"""

import base64

from gpu.build_notebooks import BUNDLED_PATHS, GET_CODE, HOW_TO_RUN, OUTPUT_DIR, build, git

ABOUT = """Each fold starts from the released checkpoint, makes one pass over that fold's training *treats* facts
with either uniform or disease-only negatives, and is scored on the held-out edges exactly as in E1."""

NOTEBOOKS = {
    "gpu_e5_ultra": (
        f"""# E5: ULTRA fine-tuned with uniform or disease-only negatives

{ABOUT} ULTRA runs on all five folds.

{HOW_TO_RUN}""",
        ['stage("e5_negatives.finetune", f"--model ultra --ultra-repo {ULTRA} --output-dir {OUTPUT}/runs")'],
    ),
    "gpu_e5_flock": (
        f"""# E5: FLOCK pilot, fine-tuned with uniform or disease-only negatives

{ABOUT} FLOCK runs on fold 0 only, because each run takes about an hour.

{HOW_TO_RUN}""",
        ['stage("e5_negatives.finetune", f"--model flock --folds 0 --output-dir {OUTPUT}/runs")'],
    ),
}


def main():
    upstream = git("rev-parse", "upstream/main").strip()
    if git("diff", "--name-only", upstream, "HEAD", "--", "src_entity", "src_relation", "graph-walker", "checkpoints"):
        raise SystemExit("upstream code differs from HEAD, so a clean upstream clone would not match this commit")
    paths = [*BUNDLED_PATHS, "experiments/e5_negatives"]
    archive = git("archive", "--format=tar.gz", "HEAD", "--", *paths, text=False)
    get_code = GET_CODE.replace("__BUNDLE__", base64.b64encode(archive).decode())
    OUTPUT_DIR.mkdir(exist_ok=True)
    commit = git("rev-parse", "HEAD").strip()
    for name, (intro, stages) in NOTEBOOKS.items():
        build(name, intro, stages, get_code, commit, upstream)


if __name__ == "__main__":
    main()
