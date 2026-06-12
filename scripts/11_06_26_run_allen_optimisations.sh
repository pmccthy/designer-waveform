#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== C1V1 optimisation ==="
python scripts/11_06_26_optimise_allen_c1v1.py

echo ""
echo "=== ChRmine optimisation ==="
python scripts/11_06_26_optimise_allen_chrmine.py

echo ""
echo "=== All done ==="
