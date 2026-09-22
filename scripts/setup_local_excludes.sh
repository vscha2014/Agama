#!/usr/bin/env bash
# Register local-only working directories in .git/info/exclude.
#
# Why not .gitignore: paths listed in .gitignore are also hidden from the AI
# agent's read/edit tools, so notes and results become unreadable. Entries in
# .git/info/exclude give the same protection against accidental `git add`
# (and are never committed), while staying readable.
#
# Run once per clone:  bash scripts/setup_local_excludes.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
EXCLUDE="$ROOT/.git/info/exclude"

add() {
    grep -qxF "$1" "$EXCLUDE" || { printf '%s\n' "$1" >> "$EXCLUDE"; echo "added: $1"; }
}

mkdir -p "$(dirname "$EXCLUDE")"
touch "$EXCLUDE"

# Run artefacts, notes and the run registry: nothing under results/ is committed
# (the repository is public and unpublished numbers must stay local).
add '/results/'
# Article drafts (also a separate private git repo).
add '/paper/'

echo "done: $EXCLUDE"
