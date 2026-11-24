"""Build the self-contained E3 GPU notebook from the code committed at HEAD.

Usage, from experiments/: uv run python -m e3_coverage.build_notebook
"""

import base64

from gpu.build_notebooks import BUNDLED_PATHS, GET_CODE, HOW_TO_RUN, OUTPUT_DIR, build, git

INTRO = f"""# E3: walk coverage on Hetionet

Runs FLOCK on a fixed sample of 200 drug → disease queries under six settings: fewer walks, shorter walks at
the same total budget, and the graph without its Anatomy *expresses* Gene edges. The reference setting comes
from the E1 notebook, on the same queries.

{HOW_TO_RUN}"""

STAGES = ['stage("e3_coverage.run_coverage", f"--output-dir {OUTPUT}/e3")']


def main():
    upstream = git("rev-parse", "upstream/main").strip()
    if git("diff", "--name-only", upstream, "HEAD", "--", "src_entity", "src_relation", "graph-walker", "checkpoints"):
        raise SystemExit("upstream code differs from HEAD, so a clean upstream clone would not match this commit")
    paths = [*BUNDLED_PATHS, "experiments/e3_coverage"]
    archive = git("archive", "--format=tar.gz", "HEAD", "--", *paths, text=False)
    get_code = GET_CODE.replace("__BUNDLE__", base64.b64encode(archive).decode())
    OUTPUT_DIR.mkdir(exist_ok=True)
    build("gpu_e3", INTRO, STAGES, get_code, git("rev-parse", "HEAD").strip(), upstream)


if __name__ == "__main__":
    main()
