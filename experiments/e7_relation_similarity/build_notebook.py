"""Build the self-contained E7 GPU notebook from the code committed at HEAD.

Usage, from experiments/: uv run python -m e7_relation_similarity.build_notebook
"""

import base64

from gpu.build_notebooks import BUNDLED_PATHS, GET_CODE, HOW_TO_RUN, OUTPUT_DIR, build, git

INTRO = f"""# E7: relation representations of ULTRA and FLOCK

Extracts each model's relation representations on Hetionet, to test whether relations that connect the
same entity types, such as *upregulates* and *downregulates*, stay distinguishable.

{HOW_TO_RUN}"""

STAGES = [
    """stage(
    "e7_relation_similarity.extract_relations",
    f"--model ultra --ultra-repo {ULTRA} --output-dir {OUTPUT}/e7",
)""",
    """stage(
    "e7_relation_similarity.extract_relations",
    f"--model flock --seeds 0,1,2 --output-dir {OUTPUT}/e7",
)""",
]


def main():
    upstream = git("rev-parse", "upstream/main").strip()
    if git("diff", "--name-only", upstream, "HEAD", "--", "src_entity", "src_relation", "graph-walker", "checkpoints"):
        raise SystemExit("upstream code differs from HEAD, so a clean upstream clone would not match this commit")
    paths = [*BUNDLED_PATHS, "experiments/e7_relation_similarity"]
    archive = git("archive", "--format=tar.gz", "HEAD", "--", *paths, text=False)
    get_code = GET_CODE.replace("__BUNDLE__", base64.b64encode(archive).decode())
    OUTPUT_DIR.mkdir(exist_ok=True)
    build("gpu_e7", INTRO, STAGES, get_code, git("rev-parse", "HEAD").strip(), upstream)


if __name__ == "__main__":
    main()
