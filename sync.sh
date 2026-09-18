#!/bin/bash
# Copies the code (not the data) to the GPU box and shows how to start a run there.
#   ./sync.sh                       -> the default host below
#   ./sync.sh user@host:/some/dir   -> anywhere else
set -euo pipefail
DEST="${1:-balbir.prasad@10.0.62.168:work/mil}"   # ramanujan, lab network
SSH_PORT="${SSH_PORT:-2022}"
cd "$(dirname "$0")"
rsync -av --delete -e "ssh -p $SSH_PORT" \
  --exclude data --exclude results --exclude .git --exclude '__pycache__' --exclude '*.ipynb_checkpoints' \
  ./ "$DEST/"
cat <<MSG

Synced to $DEST. On the server:

  cd ~/work/mil
  CUDA_VISIBLE_DEVICES=3 nohup ~/wrf/wrfEnv/bin/python train.py \
      --features data/pda-ft --out results --quiet > train.log 2>&1 &
  tail -f train.log
MSG
