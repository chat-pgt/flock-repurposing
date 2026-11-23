#!/usr/bin/env bash
# Download the Hetionet split used by the FLOCK paper into experiments/data/hetionet_authors and verify it.
set -euo pipefail

dest="$(cd "$(dirname "$0")/.." && pwd)/data/hetionet_authors"
mkdir -p "$dest"
cd "$dest"

[ -s train.txt ] || curl -fsSL -o train.txt "https://www.dropbox.com/s/y47bt9oq57h6l5k/train.txt?dl=1"
[ -s valid.txt ] || curl -fsSL -o valid.txt "https://www.dropbox.com/s/a0pbrx9tz3dgsff/valid.txt?dl=1"
[ -s test.txt ] || curl -fsSL -o test.txt "https://www.dropbox.com/s/4dhrvg3fyq5tnu4/test.txt?dl=1"

sha256sum -c <<'EOF'
cc8edc84e9e62a955affc88e21eeba7f272c04d552e53e780624c5b75ad929b2  train.txt
7c8301fb0e3635e347812111afd02580c2dd6c915dea5c4641d4aff2108f36ed  valid.txt
53afde5426cb915241887e9a7e83ba979a6bfb43a433cc25e8b552c7ec768728  test.txt
EOF
