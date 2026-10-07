#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
repo=third_party/openvla-oft
revision=e4287e94541f459edc4feabc4e181f537cd569a8
mkdir -p third_party
if [[ ! -d "$repo/.git" ]]; then
  git clone --no-checkout https://github.com/moojink/openvla-oft.git "$repo"
fi
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "Upstream reference has local changes; leaving it untouched" >&2
  exit 1
fi
git -C "$repo" fetch origin "$revision"
git -C "$repo" checkout --detach "$revision"
