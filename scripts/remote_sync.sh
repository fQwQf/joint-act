#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: $0 SSH_HOST ABSOLUTE_REMOTE_DIRECTORY" >&2
  exit 2
fi
host=$1
destination=$2
if [[ "$destination" != /* || "$destination" == / || "$destination" == *[\ \;\'\"\$\`]* ]]; then
  echo "Use a non-root absolute destination without shell metacharacters" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
rsync -az --exclude .git --exclude /third_party/ --exclude /data/ --exclude /runs/ \
  --exclude .venv --exclude __pycache__ --exclude '*.egg-info' --exclude dist \
  --exclude build --exclude .pytest_cache --exclude .ruff_cache ./ "$host:$destination/"
