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
REMOTE_DIR="/scratch/${NETID}/TRAWIC/TraWiC" # Check project name

ssh delftblue << EOF
set -euo pipefail
cd "${REMOTE_DIR}"
git pull
sbatch run_delftblue.sh
EOF

# Tun this on local machine
# Then, to see live logs on delftblue:  tail -f "$(ls -t | head -n 1)"