#!/usr/bin/env bash
# Verify CodeQL installation and run a basic query against a bundled sample.
set -euo pipefail

CODEQL_BIN="${CODEQL_BIN:-codeql}"
SKILL_DIR="${SKILL_DIR:-data/skill_dataset/messageguard}"
QUERY_DIR="${QUERY_DIR:-experiments/codeql/queries/python}"
DB_DIR="derived/codeql_db/messageguard/python"
RESULTS_DIR="derived/codeql_verify"

echo "=== SkillRecon CodeQL Verification ==="
echo ""

# 1. Check CodeQL version
echo "[1/4] CodeQL version:"
"$CODEQL_BIN" version
echo ""

# 2. Check available languages
echo "[2/4] Available language packs:"
"$CODEQL_BIN" resolve languages 2>/dev/null | head -10
echo ""

# 3. Create Python database from a bundled Python skill
echo "[3/4] Creating CodeQL database..."
mkdir -p "$(dirname "$DB_DIR")"
if [ -d "$DB_DIR" ]; then
    echo "  Database already exists at $DB_DIR, removing..."
    rm -rf "$DB_DIR"
fi

"$CODEQL_BIN" database create "$DB_DIR" \
    --language=python \
    --source-root="$SKILL_DIR" \
    --overwrite \
    2>&1 | tail -5
echo "  Database created at: $DB_DIR"
echo ""

# 4. Run the repository query pack
echo "[4/4] Running repository Python query pack..."
mkdir -p "$RESULTS_DIR"

"$CODEQL_BIN" database analyze "$DB_DIR" \
    --format=sarif-latest \
    --output="$RESULTS_DIR/basic_analysis.sarif" \
    "$QUERY_DIR" \
    2>&1 | tail -10

RESULT_COUNT=$(python3 -c "
import json
with open('$RESULTS_DIR/basic_analysis.sarif') as f:
    data = json.load(f)
runs = data.get('runs', [])
total = sum(len(r.get('results', [])) for r in runs)
print(total)
" 2>/dev/null || echo "error")

echo ""
echo "=== Verification Summary ==="
echo "  CodeQL binary:  $CODEQL_BIN"
echo "  Skill root:     $SKILL_DIR"
echo "  Query dir:      $QUERY_DIR"
echo "  Database:       $DB_DIR"
echo "  Query results:  $RESULT_COUNT findings"
echo "  SARIF output:   $RESULTS_DIR/basic_analysis.sarif"
echo ""
echo "CodeQL verification PASSED."
