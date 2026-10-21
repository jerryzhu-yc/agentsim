#!/usr/bin/env bash
set -Eeuo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
elif [[ -x venv/bin/python ]]; then
    PYTHON=venv/bin/python
else
    PYTHON=python3
fi

exec "$PYTHON" -m agentsim "$@"
