#!/bin/bash
# Push changes to remote.
# Usage: ./push.sh [commit message]
#   ./push.sh                    # uses default message
#   ./push.sh "Fix disk space"   # custom message

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MSG="${1:-Update}"
git add -A
git diff --cached --quiet && { echo "Nothing to commit."; exit 0; }
git commit -m "$MSG"
git push origin main
echo "Pushed."
