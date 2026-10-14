#!/usr/bin/env bash
# Verify this release. See checksums/README.md.
#   ./verify.sh                        -> code catalog only
#   ./verify.sh <power-aggregate-dir>  -> code catalog + power aggregates
#   ./verify.sh <power-aggregate-dir> <within-story-dir>  -> also the four within-story aggregates
#   ./verify.sh <power-aggregate-dir> <within-story-dir> <oracle-dir>  -> also the eight corrected aggregates
set -uo pipefail
cd "$(dirname "$0")"
status=0

if command -v shasum >/dev/null 2>&1; then
  hash_command=(shasum -a 256)
elif command -v sha256sum >/dev/null 2>&1; then
  hash_command=(sha256sum)
else
  echo "FAILED: neither shasum nor sha256sum is available" >&2
  exit 2
fi

echo "== released scientific source =="
if out=$(python3 scripts/build_source_manifest.py --check 2>&1); then
  echo "   $out"
else
  echo "   FAILED: $out"; status=1
fi

if [[ $# -ge 1 ]]; then
  dir=$1
  echo "== power aggregates (8 files) in $dir =="
  if [[ ! -d $dir ]]; then
    echo "   FAILED: not a directory"; status=1
  else
    cat=$PWD/checksums/power_aggregates.sha256
    if out=$(cd "$dir" && "${hash_command[@]}" -c "$cat" 2>&1); then
      echo "   OK: $(printf '%s\n' "$out" | grep -c ': OK$')/8"
    else
      echo "   FAILED:"; printf '%s\n' "$out" | grep -v ': OK$' | sed 's/^/     /'; status=1
    fi
  fi
else
  echo "== power aggregates: skipped (no directory given) =="
  echo "   pass evidence/power to check the shipped aggregates"
fi

if [[ $# -ge 2 ]]; then
  dir=$2
  echo "== within-story aggregates (4 files) in $dir =="
  if [[ ! -d $dir ]]; then
    echo "   FAILED: not a directory"; status=1
  else
    cat=$PWD/checksums/power_within_story_aggregates.sha256
    if out=$(cd "$dir" && "${hash_command[@]}" -c "$cat" 2>&1); then
      echo "   OK: $(printf '%s\n' "$out" | grep -c ': OK$')/4"
    else
      echo "   FAILED:"; printf '%s\n' "$out" | grep -v ': OK$' | sed 's/^/     /'; status=1
    fi
  fi
fi

if [[ $# -ge 3 ]]; then
  dir=$3
  echo "== corrected calibration aggregates (8 files) in $dir =="
  if [[ ! -d $dir ]]; then
    echo "   FAILED: not a directory"; status=1
  else
    cat=$PWD/checksums/power_oracle_aggregates.sha256
    if out=$(cd "$dir" && "${hash_command[@]}" -c "$cat" 2>&1); then
      echo "   OK: $(printf '%s\n' "$out" | grep -c ': OK$')/8"
    else
      echo "   FAILED:"; printf '%s\n' "$out" | grep -v ': OK$' | sed 's/^/     /'; status=1
    fi
  fi
fi

echo
if [[ $status -ne 0 ]]; then
  echo "one or more checks FAILED"
elif [[ $# -ge 1 ]]; then
  echo "source and supplied aggregate integrity checks passed"
else
  echo "source integrity passed; derived scientific evidence was not checked"
fi
exit $status
