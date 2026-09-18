#!/bin/bash
# Copies the code (not the data) to the GPU box and shows how to start a run there.
#   ./sync.sh                 -> ramanujan:~/work/mil
#   ./sync.sh host:/some/dir  -> anywhere else
set -euo pipefail
DEST="${1:-ramanujan:~/work/mil}"
cd "$(dirname "$0")"
rsync -av --delete \
  --exclude data --exclude results --exclude .git --exclude '__pycache__' --exclude '*.ipynb_checkpoints' \
  ./ "$DEST/"
cat <<MSG

Synced to $DEST. On the server:

  cd ~/work/mil
  CUDA_VISIBLE_DEVICES=3 nohup python train.py --features data/panda-phikon-features --out results --quiet > train.log 2>&1 &
  tail -f train.log
MSG
