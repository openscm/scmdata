#!/usr/bin/env bash
# Check that the licences of the project's dependencies are acceptable.
#
# liccheck unconditionally parses ./pyproject.toml with the unmaintained `toml`
# package, which cannot read our PEP 621 metadata (inline-table arrays, licence
# names containing commas). We therefore export the locked requirements and run
# liccheck from a scratch directory that contains no pyproject.toml, pointing it
# at the INI strategy file (liccheck.ini).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

uv export --no-default-groups --all-extras --no-hashes --no-emit-project > "${WORK}/requirements.txt"
cp "${ROOT}/liccheck.ini" "${WORK}/liccheck.ini"

cd "${WORK}"
uv run --no-sync --project "${ROOT}" \
    liccheck -s liccheck.ini -r requirements.txt -R "${ROOT}/licence-check.txt"
