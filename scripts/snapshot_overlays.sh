#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY_DIR="$WORKSPACE_ROOT/overlays"

snapshot_patch() {
    local repo="$1"
    local base="$2"
    local rel
    local rc

    git -C "$repo" cat-file -e "${base}^{commit}"
    git -C "$repo" diff \
        --binary --full-index --no-ext-diff --no-textconv --no-renames \
        "$base" --

    while IFS= read -r -d '' rel; do
        rc=0
        git -C "$repo" diff \
            --binary --full-index --no-ext-diff --no-textconv --no-renames \
            --no-index -- /dev/null "$rel" || rc=$?
        if [[ "$rc" -ne 1 ]]; then
            return "$rc"
        fi
    done < <(git -C "$repo" ls-files --others --exclude-standard -z)
}

snapshot_ref_diff() {
    local repo="$1"
    local base="$2"
    local target="$3"

    git -C "$repo" cat-file -e "${base}^{commit}"
    git -C "$repo" cat-file -e "${target}^{commit}"
    git -C "$repo" diff \
        --binary --full-index --no-ext-diff --no-textconv --no-renames \
        "$base" "$target" --
}

write_stable_snapshot() {
    local output="$1"
    shift
    local first
    local second
    local snapshot_tmp_root

    snapshot_tmp_root="${TMPDIR:-/tmp}"
    first="$(mktemp "$snapshot_tmp_root/vlm-overlay-first.XXXXXX")"
    second="$(mktemp "$snapshot_tmp_root/vlm-overlay-second.XXXXXX")"
    "$@" >"$first"
    "$@" >"$second"
    if ! cmp -s "$first" "$second"; then
        rm -f "$first" "$second"
        echo "Source changed while creating $output; no snapshot written." >&2
        return 1
    fi
    mv "$first" "$output"
    rm -f "$second"
}

mkdir -p "$OVERLAY_DIR"

write_stable_snapshot \
    "$OVERLAY_DIR/transformers-qwenvl-adapttoken.patch" \
    snapshot_patch \
    "$WORKSPACE_ROOT/transformers_qwenvl" \
    "e46d21afe2aaf942fd9515b2fba1d6f3affd61a6"

write_stable_snapshot \
    "$OVERLAY_DIR/lmms-eval-vatt-monitor-legacy.patch" \
    snapshot_ref_diff \
    "$WORKSPACE_ROOT/lmms-eval" \
    "c050adecbdde84fcefa8abf034986683cb97cc76" \
    "2c90ba0c80c205e2c5200de629ebd1a79ae99271"

if command -v sha256sum >/dev/null 2>&1; then
    (cd "$OVERLAY_DIR" && sha256sum ./*.patch >SHA256SUMS)
else
    (cd "$OVERLAY_DIR" && shasum -a 256 ./*.patch >SHA256SUMS)
fi

echo "Wrote deterministic overlays and $OVERLAY_DIR/SHA256SUMS"
