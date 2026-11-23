"""Record what produced a result (PROTOCOL.md, section 3)."""

import hashlib
import platform
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_metadata(data_files=(), **settings):
    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "code_modified_since_commit": bool(git("status", "--porcelain", "--", "experiments", ":!experiments/results")),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "data_sha256": {Path(path).name: sha256(path) for path in data_files},
        **settings,
    }
