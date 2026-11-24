"""Build the self-contained E4 GPU notebook from the code committed at HEAD.

Usage, from experiments/: uv run python -m e4_explanations.build_notebook
"""

import base64

from gpu.build_notebooks import BUNDLED_PATHS, GET_CODE, HOW_TO_RUN, OUTPUT_DIR, build, git

INTRO = f"""# E4: which walks carry FLOCK's predictions, and do they matter

Scores four showcase drug → disease pairs (three walk seeds) and a survey of 40 held-out edges (one seed)
zero-shot with FLOCK's entity checkpoint, each on the fold where its edge is held out. It credits each walk
with its share of the disease's state, names the drug → disease stretches, and rescores after deleting the
top stretches' facts, random facts, and families of the drug's facts. About 40 minutes on a Kaggle T4.

{HOW_TO_RUN}"""

STAGES = ['stage("e4_explanations.explain_walks", f"--output-dir {OUTPUT}/e4")']
SUMMARY = """sh(f"grep -F ' (fold ' {OUTPUT}/{NOTEBOOK}.log > {OUTPUT}/{NOTEBOOK}_summary.txt", check=False)"""


def main():
    upstream = git("rev-parse", "upstream/main").strip()
    if git("diff", "--name-only", upstream, "HEAD", "--", "src_entity", "src_relation", "graph-walker", "checkpoints"):
        raise SystemExit("upstream code differs from HEAD, so a clean upstream clone would not match this commit")
    paths = [*BUNDLED_PATHS, "experiments/e4_explanations"]
    archive = git("archive", "--format=tar.gz", "HEAD", "--", *paths, text=False)
    get_code = GET_CODE.replace("__BUNDLE__", base64.b64encode(archive).decode())
    OUTPUT_DIR.mkdir(exist_ok=True)
    build("gpu_e4", INTRO, STAGES, get_code, git("rev-parse", "HEAD").strip(), upstream, summary=SUMMARY)


if __name__ == "__main__":
    main()
