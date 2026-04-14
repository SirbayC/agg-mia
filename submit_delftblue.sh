#!/usr/bin/env bash
set -euo pipefail

if [[ -n "$(git status --porcelain)" ]]; then
	git add .
	git commit -m "$(date +%Y%m%d_%H%M%S)"
	git push
else
	echo "No local changes; skipping commit/push."
fi

NETID="cosminvasilesc"
REMOTE_DIR="/scratch/${NETID}/AGG_MIA/agg-mia" # Check project name

ssh delftblue << EOF
set -euo pipefail
cd "${REMOTE_DIR}"
git pull
sbatch run_delftblue.sh
EOF
