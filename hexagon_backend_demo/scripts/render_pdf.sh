#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CONFIG=${HEXAGON_DEMO_CONFIG:-$BUNDLE_ROOT/config.sh}
[[ -f "$CONFIG" ]] && source "$CONFIG"

PYTHON=${TILELANG_PYTHON:-python3}
if ! "$PYTHON" -c 'import weasyprint' >/dev/null 2>&1; then
    echo "WeasyPrint is missing for $PYTHON" >&2
    echo "Install: $PYTHON -m pip install -r '$BUNDLE_ROOT/requirements-presentation.txt'" >&2
    exit 1
fi

"$PYTHON" -m weasyprint --quiet \
    "$BUNDLE_ROOT/index.html" \
    "$BUNDLE_ROOT/hexagon_backend_technical_route.pdf"

if command -v pdfinfo >/dev/null 2>&1; then
    pdfinfo "$BUNDLE_ROOT/hexagon_backend_technical_route.pdf" | \
        awk '/^(Pages|Page size|File size):/'
fi
