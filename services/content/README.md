# content

This folder has the ZIM catalog and the downloader for the hub's offline knowledge library. kiwix-serve serves these ZIMs for browsing, and `ingest_zim.py` embeds them into LanceDB.

## Files

- `zim-catalog.conf` lists every ZIM the hub carries. For each one it records the Kiwix folder, a tier, an embed policy (`full`, `selective` or `none`) and a kind (`html` or `pdf`).
- `fetch-zims.sh` finds the newest dated build of each entry, downloads it into `$ZIM_DIR`, checks the `.sha256` when Kiwix publishes one, writes `manifest.tsv`, and rebuilds `library.xml` for kiwix-serve.

## Usage

```bash
# See what would be downloaded. Fast, and nothing is written.
ZIM_DIR=/mnt/content/zim services/content/fetch-zims.sh --check

# Download core content first, then everything else.
ZIM_DIR=/mnt/content/zim services/content/fetch-zims.sh --tier 1
ZIM_DIR=/mnt/content/zim services/content/fetch-zims.sh

# Serve the library.
kiwix-serve --port 8888 --library /mnt/content/zim/library.xml
```

You can re-run it whenever you like. Files that are already downloaded get skipped, and a partial download picks up where it stopped. Older versions of a ZIM stay on disk unless you pass `--prune`. After pruning, re-run `ingest_zim.py` for that ZIM so LanceDB doesn't keep pointing at a file that's gone.

`kiwix-manage` and `kiwix-serve` come from kiwix-tools. Kiwix publishes static `linux-aarch64` builds that run on the Rock 5C.

## Notes for ingest_zim.py

- Read `$ZIM_DIR/manifest.tsv` to decide what to embed. Skip anything marked `embed=none`, and apply the category or tag filters to anything marked `selective`.
- The `kind=pdf` ZIMs (the `zimgit-*` ones) are nautilus collections, meaning each one is a set of PDFs plus a small JS viewer. They need PDF text extraction, because H2/H3 chunking won't find anything in them. In the browser, these collections keep their book list in IndexedDB, so they won't load in iOS private tabs or the captive-portal popup.
- `mdwiki_en_all_maxi` is set to browse-only because it shares a source with WikiMed (`wikipedia_en_medicine_maxi`). Embedding both would duplicate chunks.

## Not yet verified

The cloud session that wrote this couldn't reach download.kiwix.org, so some folder and prefix pairs haven't been checked: the LibreTexts entries, khanacademy and crashcourse. Run `--check` once on a machine that has internet access, then fix any line it reports as `not found`.
