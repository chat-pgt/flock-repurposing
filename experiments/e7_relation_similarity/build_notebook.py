"""Build the self-contained E7 GPU notebook from the code committed at HEAD.

Usage, from experiments/: uv run python -m e7_relation_similarity.build_notebook
"""

import argparse
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


FOLLOWUP_INTRO = f"""# E7 follow-up: can the models tell same-type relations apart?

Runs, in priority order:

1. FLOCK's relation states with a neutral query relation (*Compound causes Side Effect*), so that *treats*
   and *palliates* can be compared without the query marker.
2. ULTRA and FLOCK as relation predictors on 100 *upregulates* and 100 *downregulates* Compound–Gene test
   triples. The sample is balanced, so chance is 50%.

{HOW_TO_RUN}"""

FOLLOWUP_STAGES = [
    """stage(
    "e7_relation_similarity.extract_relations",
    f"--model flock --seeds 0,1,2 --query-relation CcSE --output-dir {OUTPUT}/e7",
)""",
    """stage(
    "e7_relation_similarity.run_relation_pairs",
    f"--model ultra --ultra-repo {ULTRA} --output-dir {OUTPUT}/e7",
)""",
    """stage(
    "e7_relation_similarity.run_relation_pairs",
    f"--model flock --seeds 0,1,2 --output-dir {OUTPUT}/e7",
)""",
]
NOTEBOOKS = {"gpu_e7": (INTRO, STAGES), "gpu_e7_followup": (FOLLOWUP_INTRO, FOLLOWUP_STAGES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", choices=[[], *NOTEBOOKS], help="Notebooks to build (default: all)")
    args = parser.parse_args()
    upstream = git("rev-parse", "upstream/main").strip()
    if git("diff", "--name-only", upstream, "HEAD", "--", "src_entity", "src_relation", "graph-walker", "checkpoints"):
        raise SystemExit("upstream code differs from HEAD, so a clean upstream clone would not match this commit")
    paths = [*BUNDLED_PATHS, "experiments/e7_relation_similarity"]
    archive = git("archive", "--format=tar.gz", "HEAD", "--", *paths, text=False)
    get_code = GET_CODE.replace("__BUNDLE__", base64.b64encode(archive).decode())
    OUTPUT_DIR.mkdir(exist_ok=True)
    commit = git("rev-parse", "HEAD").strip()
    for name in args.names or NOTEBOOKS:
        intro, stages = NOTEBOOKS[name]
        build(name, intro, stages, get_code, commit, upstream)


if __name__ == "__main__":
    main()
