"""Build the self-contained E4 GPU notebook from the code committed at HEAD.

Usage, from experiments/: uv run python -m e4_explanations.build_notebook
"""

import base64

from gpu.build_notebooks import BUNDLED_PATHS, GET_CODE, HOW_TO_RUN, OUTPUT_DIR, build, git

INTRO = f"""# E4: which walks carry FLOCK's prediction

Scores three held-out drug → disease pairs zero-shot with FLOCK's entity checkpoint, on the fold where each is
held out, with three walk seeds. It records how much each walk contributes to the target disease's state and
maps the walks back to entity and relation names. It takes a few minutes, so Colab is enough.

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
