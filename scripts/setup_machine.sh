#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_KEY="${GITHUB_SSH_KEY:-${1:-}}"

if [[ -z "$SSH_KEY" || "$SSH_KEY" == "-h" || "$SSH_KEY" == "--help" ]]; then
    cat <<'EOF'
Usage: scripts/setup_machine.sh /path/to/github-private-key

Installs Homebrew tools, persists the SSH/tool shell environment, updates the
public repository, reconstructs pinned checkouts, and runs ./install.sh.
The key can alternatively be supplied with GITHUB_SSH_KEY.
EOF
    [[ -n "$SSH_KEY" ]] && exit 0
    exit 2
fi

[[ -f "$SSH_KEY" ]] || {
    echo "SSH key not found: $SSH_KEY" >&2
    exit 2
}
SSH_KEY_DIR="$(cd "$(dirname "$SSH_KEY")" && pwd -P)"
SSH_KEY="$SSH_KEY_DIR/$(basename "$SSH_KEY")"
chmod 600 "$SSH_KEY"
export GIT_SSH_COMMAND="ssh -i \"$SSH_KEY\" -o IdentitiesOnly=yes"

if ! command -v brew >/dev/null 2>&1; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
elif [[ -x /home/linuxbrew/.linuxbrew/bin/brew ]]; then
    eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
fi

brew install git gh uv dust fzf ripgrep fd

SHELL_CONFIG_DIR="$HOME/.config/vlmbehave"
SHELL_CONFIG="$SHELL_CONFIG_DIR/shell.sh"
mkdir -p "$SHELL_CONFIG_DIR"

ESCAPED_ROOT="$(printf '%s' "$WORKSPACE_ROOT" | sed 's/[&|\\]/\\&/g')"
ESCAPED_KEY="$(printf '%s' "$SSH_KEY" | sed 's/[&|\\]/\\&/g')"
sed \
    -e "s|__VLM_WORKSPACE__|$ESCAPED_ROOT|g" \
    -e "s|__GITHUB_SSH_KEY__|$ESCAPED_KEY|g" \
    "$WORKSPACE_ROOT/config/vlmbehave-shell.sh.template" >"$SHELL_CONFIG"

case "$(basename "${SHELL:-bash}")" in
    zsh) RC_FILE="$HOME/.zshrc" ;;
    *) RC_FILE="$HOME/.bashrc" ;;
esac
SOURCE_LINE='[ -f "$HOME/.config/vlmbehave/shell.sh" ] && source "$HOME/.config/vlmbehave/shell.sh"'
touch "$RC_FILE"
grep -Fqx "$SOURCE_LINE" "$RC_FILE" || printf '\n%s\n' "$SOURCE_LINE" >>"$RC_FILE"

git -C "$WORKSPACE_ROOT" remote set-url origin https://github.com/parsa-ra/lmbehave.git
git -C "$WORKSPACE_ROOT" pull --ff-only

python3 "$WORKSPACE_ROOT/scripts/bootstrap_workspace.py"
"$WORKSPACE_ROOT/install.sh"

printf '\nSetup complete. Open a new shell, then run: vlm_activate\n'
