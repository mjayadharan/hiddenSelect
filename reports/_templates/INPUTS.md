# R### — inputs

All inputs are inside this folder. Nothing is read from another report at
run time.

## Copied into `external_data/` (small, versioned here)

| file | origin | used for |
|---|---|---|
| `<file>.csv` | <report/producer> | <what it feeds> |

## Manifested only (large; stays at its committed location)

| path | size | tracked files | produced by | hash |
|---|---|---|---|---|

## Frozen code snapshot

`deps/` is a byte copy of <source> at <commit>, minus caches; every script
puts `deps/` first on `sys.path`. See `deps/MANIFEST.md`.

## Hash ledger

`analysis/results/results_hashes.json` records SHA-256 for every input and
output CSV, written by the verify script.
