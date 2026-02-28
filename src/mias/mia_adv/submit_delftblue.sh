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
REMOTE_DIR="/scratch/${NETID}/MIA-ADV/MIA_Adv"

ssh delftblue << EOF
set -euo pipefail
cd "${REMOTE_DIR}"
git pull
sbatch run_delftblue.sh
EOF

# RUN THIS LOCALLY!
# Then, to see logs:  tail -f "$(ls -t | head -n 1)"