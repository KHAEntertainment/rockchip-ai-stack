#!/usr/bin/env bash
# Download the newest build of every ZIM in zim-catalog.conf and rebuild the
# kiwix-serve library. Safe to re-run: finished files are skipped and partial
# downloads resume.
#
# Kiwix publishes ZIMs with a date suffix (wikipedia_en_all_maxi_2026-10.zim),
# so hardcoded URLs go stale. This script resolves the newest file for each
# prefix from the folder listing instead. Pattern adapted from
# github.com/schematrix/HAS_Prepper_Disk (MIT), without the Internet-in-a-Box
# dependency.
#
# Usage:
#   fetch-zims.sh [--tier N] [--check] [--prune] [--conf FILE]
#     --tier N   only entries with tier <= N (default 3 = everything)
#     --check    resolve and print newest file names, download nothing
#     --prune    delete older versions of a ZIM once its newer one is in place
#                (re-run ingest_zim.py for that ZIM afterwards)
#
# Environment:
#   ZIM_DIR       where ZIMs live (default /mnt/content/zim)
#   KIWIX_BASE    mirror base URL (default https://download.kiwix.org/zim)
#   KIWIX_MANAGE  kiwix-manage binary (default: kiwix-manage on PATH)
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONF="$HERE/zim-catalog.conf"
ZIM_DIR="${ZIM_DIR:-/mnt/content/zim}"
BASE="${KIWIX_BASE:-https://download.kiwix.org/zim}"
KIWIX_MANAGE="${KIWIX_MANAGE:-kiwix-manage}"
MAX_TIER=3
CHECK=0
PRUNE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --tier)
      case "${2:-}" in ''|*[!0-9]*) echo "--tier needs a whole number, got: '${2:-}'" >&2; exit 2 ;; esac
      MAX_TIER="$2"; shift 2 ;;
    --check) CHECK=1; shift ;;
    --prune) PRUNE=1; shift ;;
    --conf)  CONF="$2"; shift 2 ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -f "$CONF" ] || { echo "catalog not found: $CONF" >&2; exit 2; }
if [ "$CHECK" -eq 0 ]; then
  mkdir -p "$ZIM_DIR" || exit 1
  # One writer per ZIM_DIR: runs share the manifest, .part files and library.xml.
  exec 9>"$ZIM_DIR/.fetch-zims.lock"
  flock -n 9 || { echo "!! another fetch-zims.sh run is active in $ZIM_DIR" >&2; exit 1; }
fi

LISTINGS="$(mktemp -d)"
SUPERSEDED="$LISTINGS/superseded"   # files to delete once the new library is live
: > "$SUPERSEDED"
trap 'rm -rf "$LISTINGS"' EXIT

# Folder listing, fetched once per folder.
listing() {
  local folder="$1" f="$LISTINGS/$1.html"
  [ -s "$f" ] || curl -fsSL "$BASE/$folder/" -o "$f" || { rm -f "$f"; return 1; }
  cat "$f"
}

# Newest file name for a prefix, e.g. wikem_en_all_maxi -> wikem_en_all_maxi_2026-09.zim
newest() {
  local folder="$1" prefix="$2" re
  re="$(printf '%s' "$prefix" | sed 's/[.[\*^$+?(){}|]/\\&/g')"
  listing "$folder" | grep -oE "${re}_[0-9]{4}-[0-9]{2}[a-z]?\.zim" | sort -uV | tail -1
}

# Local dated builds of a prefix (never matches e.g. prefix_custom.zim).
local_builds() {
  local prefix="$1" f
  for f in "$ZIM_DIR/${prefix}"_[0-9][0-9][0-9][0-9]-[0-9][0-9].zim \
           "$ZIM_DIR/${prefix}"_[0-9][0-9][0-9][0-9]-[0-9][0-9][a-z].zim; do
    [ -f "$f" ] && printf '%s\n' "$f"
  done | sort -V
}

failed=0
manifest="$ZIM_DIR/manifest.tsv.new"
[ "$CHECK" -eq 1 ] || printf 'file\tprefix\ttier\tembed\tkind\n' > "$manifest"

while read -r folder prefix tier embed kind _ || [ -n "${folder:-}" ]; do
  case "$folder" in ''|\#*) continue ;; esac
  if ! [ "$tier" -le "$MAX_TIER" ] 2>/dev/null; then
    # Out of scope for this run, but keep it in the manifest if it's on disk.
    if [ "$CHECK" -eq 0 ]; then
      have="$(local_builds "$prefix" | tail -1)"
      [ -n "$have" ] && printf '%s\t%s\t%s\t%s\t%s\n' "$(basename "$have")" "$prefix" "$tier" "$embed" "$kind" >> "$manifest"
    fi
    continue
  fi

  latest="$(newest "$folder" "$prefix")"
  if [ -z "$latest" ]; then
    echo "!! $prefix: not found in $BASE/$folder/"
    failed=1; continue
  fi

  if [ "$CHECK" -eq 1 ]; then
    printf 'ok  %-45s %s/%s\n' "$prefix" "$folder" "$latest"
    continue
  fi

  dest="$ZIM_DIR/$latest"
  if [ -f "$dest" ]; then
    echo "== $latest already present"
  else
    echo ">> downloading $latest"
    df -h "$ZIM_DIR" | tail -1
    if ! wget -c --progress=dot:giga -O "$dest.part" "$BASE/$folder/$latest" < /dev/null; then
      echo "!! $latest download failed (re-run to resume)"
      failed=1; continue
    fi
    # Kiwix publishes a .sha256 next to most files. Only a 404 means "none
    # published"; any other failure fails this file rather than skipping the check.
    code="$(curl -sSL -o "$dest.sha256" -w '%{http_code}' "$BASE/$folder/$latest.sha256")" || code=000
    case "$code" in
      200)
        sum="$(awk '{print $1}' "$dest.sha256")"
        if [ -z "$sum" ] || ! echo "$sum  $dest.part" | sha256sum -c --status; then
          echo "!! $latest checksum mismatch, removing partial file"
          rm -f "$dest.part" "$dest.sha256"; failed=1; continue
        fi ;;
      404) ;;
      *)
        echo "!! $latest checksum request failed (HTTP $code); keeping .part, re-run to retry"
        rm -f "$dest.sha256"; failed=1; continue ;;
    esac
    rm -f "$dest.sha256"
    mv "$dest.part" "$dest"
    echo "OK $latest"
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' "$latest" "$prefix" "$tier" "$embed" "$kind" >> "$manifest"

  while read -r old; do
    [ -n "$old" ] && [ "$old" != "$dest" ] || continue
    # Never touch a local build that sorts newer than what the mirror offers.
    [ "$(printf '%s\n%s\n' "$(basename "$old")" "$latest" | sort -V | tail -1)" = "$latest" ] || continue
    if [ "$PRUNE" -eq 1 ]; then
      echo "$old" >> "$SUPERSEDED"
    else
      echo "   note: older $(basename "$old") kept (use --prune to remove)"
    fi
  done < <(local_builds "$prefix")
done < "$CONF"

[ "$CHECK" -eq 1 ] && exit $failed

mv "$manifest" "$ZIM_DIR/manifest.tsv"

# Rebuild the kiwix-serve library from what's on disk, leaving out files about
# to be pruned. Publish only if every add succeeded, and prune only after that,
# so library.xml never points at a deleted file.
if command -v "$KIWIX_MANAGE" >/dev/null; then
  lib="$ZIM_DIR/library.xml"
  rm -f "$lib.new"
  adds_ok=1
  for z in "$ZIM_DIR"/*.zim; do
    [ -f "$z" ] || continue
    grep -qxF "$z" "$SUPERSEDED" && continue
    "$KIWIX_MANAGE" "$lib.new" add "$z" || { echo "!! $(basename "$z") not added to library"; adds_ok=0; }
  done
  if [ "$adds_ok" -eq 1 ] && [ -f "$lib.new" ] && mv "$lib.new" "$lib"; then
    echo ">> rebuilt $lib"
    while read -r old; do
      [ -n "$old" ] && echo "-- removing superseded $(basename "$old")" && rm -f "$old"
    done < "$SUPERSEDED"
  else
    rm -f "$lib.new"
    echo "!! library.xml left unchanged; nothing pruned"
    failed=1
  fi
else
  echo "!! $KIWIX_MANAGE not found; library.xml not rebuilt, nothing pruned (install kiwix-tools)"
  failed=1
fi

exit $failed
