#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
TILELANG_ROOT=${TILELANG_ROOT:-$(cd "$BUNDLE_ROOT/.." && pwd)}
LLAMA_CPP_ROOT=${LLAMA_CPP_ROOT:-/home/xwh/scratchpad-llamacpp}

copy_list() {
    local source_root=$1
    local list_file=$2
    local destination=$3
    local entry

    while IFS= read -r entry || [[ -n "$entry" ]]; do
        [[ -z "$entry" || "$entry" == \#* ]] && continue
        if [[ ! -e "$source_root/$entry" ]]; then
            printf 'missing: %s\n' "$source_root/$entry" >&2
            return 1
        fi
        if [[ -d "$source_root/$entry" ]]; then
            mkdir -p "$destination/$entry"
            rsync -a --delete --delete-excluded \
                --exclude '__pycache__/' --exclude '*.pyc' \
                "$source_root/$entry/" "$destination/$entry/"
        else
            mkdir -p "$destination/$(dirname "$entry")"
            cp -a "$source_root/$entry" "$destination/$entry"
        fi
    done < "$list_file"
}

copy_list "$TILELANG_ROOT" "$BUNDLE_ROOT/source_paths.txt" "$BUNDLE_ROOT/sources/tilelang"
copy_list "$LLAMA_CPP_ROOT" "$BUNDLE_ROOT/llama_source_paths.txt" "$BUNDLE_ROOT/sources/llama.cpp"

(
    cd "$BUNDLE_ROOT/sources"
    find tilelang llama.cpp -type f -print0 \
        | sort -z \
        | xargs -0 sha256sum \
        > SOURCE_SHA256SUMS.txt
)

printf 'refreshed source snapshot\n'
printf '  TileLang: %s\n' "$TILELANG_ROOT"
printf '  llama.cpp: %s\n' "$LLAMA_CPP_ROOT"
