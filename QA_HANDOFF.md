# QA Handoff — embedlab

This file explains how to set up the dev environment and run the tests and CLI flows for quick QA verification. It also documents optional features (ANN / PCA) that require additional system/tooling steps.

## Goals

- Verify unit tests pass.
- Verify the CLI `embed`, `search`, and `analyze` commands run end-to-end on a small image set.
- Validate optional Annoy-based ANN search (if you install Annoy).
- Confirm index save/load compatibility and the presence of `embeddings.npy`, `paths.json`, `meta.json`, and `manifest.json`.

## Environment (macOS, zsh)

1. Create and activate the virtualenv (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

1. Upgrade pip and install the pinned dev requirements (core only):

```bash
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements-dev.txt
```

Notes:

- `requirements-dev.txt` intentionally does NOT pin `annoy` or `scikit-learn` because they may require a C/C++ toolchain on macOS and certain Python versions.
- If you need ANN (Annoy) or PCA/metrics (scikit-learn), install them manually after the above step:

```bash
# Optional: install Annoy and scikit-learn (may require Xcode Command Line Tools)
.venv/bin/python -m pip install annoy scikit-learn
```

If pip fails building wheels on macOS, ensure Xcode Command Line Tools are installed:

```bash
xcode-select --install
```

## Run unit tests

From the repository root:

```bash
.venv/bin/pytest -q
```

Expected: all tests pass (currently 3 small deterministic tests). If tests fail, run with `-q -k <testname>` to debug a single test.

## Quick end-to-end QA (small image set)

1. Prepare test images

- Create a tiny dataset: a folder `assets/images` with 3–10 JPEG/PNG images. You can reuse existing test image fixtures or generate small images with Python/Pillow.

1. Embed images (write an index)

```bash
# embedlab.py is the script entrypoint
python embedlab.py embed --images-dir ./assets/images --out ./index --batch-size 8 --seed 42
```

After completion, the `./index` directory should contain:

- `embeddings.npy` (float32 2D array)
- `paths.json` (list of image file paths)
- `meta.json` (backbone, dim, image_count, created_at, seed)
- `manifest.json` (performance and reproducibility info)


1. Run search (exact)

```bash
python embedlab.py search --index ./index --query-dir ./assets/queries --k 5
```

Expected: JSON or pretty results listing top-k matches with scores. For small test images, results should be sensible based on visual similarity.


1. Run analyze (duplicates & anomalies)

```bash
python embedlab.py analyze --index ./index --dup-threshold 0.95 --anomaly-top 5 --knn 5
```

Expected: CLI prints or writes a summary including `duplicate_groups` and `anomalies`. The union-find k-NN path will be used when `knn` is provided.

## Testing ANN (optional)

If you installed `annoy`, test the ANN path with the `--ann` flag and tune `--n-trees`:

```bash
python embedlab.py search --index ./index --query-dir ./assets/queries --k 5 --ann --n-trees 32
```

Verify the output is similar to exact search (for small datasets they should match) and note the printed timing info when `--ann` is used.

Benchmark idea (recall@5):

- Create a small validation set and run both exact and ANN searches. Compare recall@5 (fraction of exact-top5 present in ANN top5). For larger datasets use the provided index and measure recall/time tradeoffs.

## Notes about optional advanced features

- Compression (PCA -> 128D) and FP16 storage are currently planned but not yet implemented. If you implement or enable them, re-run tests and add integration tests that assert `embeddings.npy` shape and dtype.
- Calibration (learning a similarity threshold from a labeled validation split) is not yet implemented. If you add it, include a small unit test with a deterministic toy dataset that validates ROC/PR outputs.

## Troubleshooting

- If tests fail due to missing packages, check which optional packages your local flow requires and install them manually.
- On macOS, building wheels for packages like `annoy` and `scikit-learn` requires a working C/C++ toolchain; run `xcode-select --install` and consider using a Python version with prebuilt wheels.
- If `embeddings.npy` cannot be memory-mapped, ensure it was saved with dtype `float32` and a 2D shape.

## Files to inspect when something goes wrong

- `src/embedlib.py` — core logic for encoding, saving/loading index, search, and analyze.
- `src/embedlab_cli.py` — CLI glue and flag parsing.
- `tests/` — unit tests used by CI and local QA.

## CI

CI runs lint, type-check (mypy), and pytest across supported Python versions. If you change dependencies (add Annoy/scikit-learn), consider adding a separate CI job that prepares the environment with the required toolchain or pins Python versions that have prebuilt wheels.

## Contact / Handoff notes

If you hit a blocker while testing (build failures, failing tests after your change), include:

- OS / Python version
- Output of the failing command (attach the full pytest output)
- Whether you installed optional deps (`annoy`, `scikit-learn`)

That's it — this file should give QA the steps they need to validate core functionality and understand optional features that need additional host toolchain support.
