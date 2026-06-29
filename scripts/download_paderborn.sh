#!/usr/bin/env bash
# Download Paderborn University bearing dataset (KAt) into data/raw/paderborn_pu/.
#
# License: academic use only. By running this script you confirm you accept
# the dataset's terms of use at https://groups.uni-paderborn.de/kat/BearingDataCenter/
#
# Total size: ~12 GB compressed (RAR), ~36 GB uncompressed.
#
# The KAt site distributes per-bearing RAR archives. We fetch a curated subset
# matching our 4-class mapping (see src/phm_routing/data/paderborn.py).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/raw/paderborn_pu"
mkdir -p "${DEST}"
cd "${DEST}"

BASE_URL="https://groups.uni-paderborn.de/kat/BearingDataCenter"

# 4-class subset:
#   N (healthy):     K001..K006
#   IR (inner):      KI01..KI09 (artificial) + KI14, KI16-18, KI21 (real)
#   OR (outer):      KA01..KA09 (artificial) + KA15-22, KA30 (real)
#   RE (combined):   KB23, KB24, KB27 (real)
BEARINGS=(
  K001 K002 K003 K004 K005 K006
  KI01 KI03 KI04 KI05 KI07 KI08 KI14 KI16 KI17 KI18 KI21
  KA01 KA03 KA04 KA05 KA07 KA08 KA15 KA16 KA22 KA30
  KB23 KB24 KB27
)

if ! command -v unrar >/dev/null 2>&1 && ! command -v 7z >/dev/null 2>&1; then
  echo "ERROR: need 'unrar' or '7z' installed to extract RAR archives." >&2
  echo "  sudo apt-get install -y unrar    # or:    sudo apt-get install -y p7zip-full" >&2
  exit 1
fi

extract() {
  local rar="$1"
  if command -v unrar >/dev/null 2>&1; then
    unrar x -inul -o+ "${rar}"
  else
    7z x -y "${rar}" >/dev/null
  fi
}

for B in "${BEARINGS[@]}"; do
  if [[ -d "${B}" ]]; then
    echo "[skip] ${B} already extracted"
    continue
  fi
  RAR="${B}.rar"
  if [[ ! -f "${RAR}" ]]; then
    echo "[fetch] ${B}"
    if ! curl -fL -o "${RAR}" "${BASE_URL}/${RAR}"; then
      echo "  WARN: ${B} not available at primary URL, skipping" >&2
      rm -f "${RAR}"
      continue
    fi
  fi
  echo "[extract] ${B}"
  extract "${RAR}" || { echo "  WARN: extract failed for ${B}" >&2; continue; }
done

echo
echo "Done. Files:"
find "${DEST}" -name '*.mat' | wc -l
