#!/usr/bin/env bash
# Download the official Hetionet v1.0 files into experiments/data/hetionet and verify them.
set -euo pipefail

dest="$(cd "$(dirname "$0")/.." && pwd)/data/hetionet"
base="https://github.com/hetio/hetionet/raw/main/hetnet/tsv"
mkdir -p "$dest"
cd "$dest"

for file in hetionet-v1.0-edges.sif.gz hetionet-v1.0-nodes.tsv; do
  [ -f "$file" ] || curl -fsSL -o "$file" "$base/$file"
done

sha256sum -c <<'EOF'
611f0411ac666be4e0270e54bbca67407e14720bfc3099311f0f5eec25c5a947  hetionet-v1.0-edges.sif.gz
3ad9eb86c031df8bc7d5efefa6edc93562182eb9a6a71e5129d3f18b9d42c50e  hetionet-v1.0-nodes.tsv
EOF
