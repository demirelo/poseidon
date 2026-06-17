# paper/ - manuscript

This directory contains the paper source, built PDF, and bibliography for:

*Concrete-Complexity Cryptanalysis of the 2026 Poseidon1/KoalaBear Challenge
Suite*

The paper is a reduced-round and concrete-complexity study. It does not claim a
full-round RF=6 solve.

## Files

- `poseidon2026-kb.tex`: canonical manuscript source.
- `poseidon2026-kb.pdf`: built PDF.
- `refs.bib`: checked bibliography metadata.

## Submission mode

- `\submissiontrue` is enabled by default.
- `[V]` margin tags are blank in the submission PDF.
- Any `\TODO{...}` hard-fails compilation.

## Build

```sh
cd paper && pdflatex poseidon2026-kb.tex && pdflatex poseidon2026-kb.tex
# self-contained (inline thebibliography); the 2nd pass resolves cross-refs. No bibtex needed.
```

Builds clean with MacTeX `pdflatex` (2 passes, no errors / no undefined refs or
citations) to an **8-page** PDF, committed as
[`poseidon2026-kb.pdf`](poseidon2026-kb.pdf). Uses only standard packages
(geometry, amsmath, booktabs, hyperref, xcolor); CI also rebuilds
the PDF on every push (`.github/workflows/ci.yml`).

## Verification

The reduced-round RF=4/RP=5 witness used by the paper passes the independent
checker and the pinned official reference verifier:

```sh
python3 ../submission/verify_with_official.py \
  ../submission/examples/good_zerotest_rf4_rp5.json --official
```

Durable JSON artifacts referenced by the paper live under
[`../artifacts/`](../artifacts/).
