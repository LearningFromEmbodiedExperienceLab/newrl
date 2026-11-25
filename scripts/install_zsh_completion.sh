#!/bin/sh
# Install zsh completion for NewRL commands and config presets.
# Activate the desired project environment before running this installer.

set -eu

root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
zshrc=${ZSHRC:-"$HOME/.zshrc"}
dest=${NEWRL_COMPLETION:-"$HOME/.local/share/newrl/newrl-completion.zsh"}

mkdir -p "$(dirname "$dest")"
if [ -n "${VIRTUAL_ENV:-}" ]; then
  set -- "$VIRTUAL_ENV/bin/python"
else
  set -- "$root/.venv/bin/python"
fi
if [ ! -x "$1" ]; then
  echo "Activate an environment with NewRL installed before running this script." >&2
  exit 1
fi
tmp=$(mktemp "$dest.XXXXXX")
trap 'rm -f "$tmp"' EXIT
"$1" "$root/scripts/generate_zsh_completion.py" > "$tmp"
mv "$tmp" "$dest"

marker="# newrl-completion"
quoted=$(printf "%s" "$dest" | sed "s/'/'\\\\''/g")
source_line="$marker
source '$quoted'"
if [ -f "$zshrc" ] && grep -q "$marker" "$zshrc"; then
    echo "Already installed in $zshrc"
else
    touch "$zshrc"
    printf '\n%s\n' "$source_line" >> "$zshrc"
    echo "Added completion to $zshrc"
fi
echo "Wrote $dest"
echo "Open a new zsh, or run: source '$zshrc'"
