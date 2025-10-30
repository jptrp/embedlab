# embedlab — Image Embeddings, Search, Dedupe & Anomaly Detection (CLI)

CLI tool that:
- **Embeds** images into fixed-length vectors using a reasonable vision backbone (default: torchvision ResNet18/IMAGENET1K).
- **Searches** nearest neighbors for query images with cosine similarity (exact or ANN).
- **Analyzes** near-duplicates and anomalies using only embeddings.

## Requirements
- Python 3.10+
- Runtime deps for `embed`: `torch`, `torchvision`, `Pillow`, `numpy`

Optional (recommended for advanced workflows):
- `annoy` — for fast approximate nearest-neighbor (ANN) indexing and tiered search
- `scikit-learn` — utilities such as PCA for compression/experiments
- `CLIP` or other text-image encoders — for zero-shot text queries

Note: some optional packages (Annoy, scikit-learn) may require a C/C++ toolchain on macOS and Linux. If you hit build errors, install Xcode command line tools or use prebuilt wheels where available.

## CLI

The CLI exposes three main subcommands: `embed`, `search`, and `analyze`.

### Embed
Create an index of embeddings from a directory of images.

Basic:

```bash
python embedlab.py embed --images-dir ./assets/images --out ./index
```

Useful flags:
- `--batch-size N` — encode images in batches (improves throughput)
- `--seed N` — make embedding deterministic
- `--no-metrics` — do not write `metrics.json` to the output index

### Search
Search nearest neighbors for query images. Supports exact cosine search or an ANN-backed tiered flow (ANN candidates → exact rerank).

Basic exact search:

```bash
python embedlab.py search --index ./index --query-dir ./assets/queries --k 5 --json
```

ANN / tiered examples:

```bash
# Use Annoy for ANN (must have `annoy` installed)
python embedlab.py search --index ./index --query-dir ./assets/queries --k 5 --ann --n-trees 10

# Tiered: ANN candidates then exact rerank (recommended for speed/quality tradeoff)
python embedlab.py search --index ./index --query-dir ./assets/queries --k 5 --ann --tiered --candidates 100

# Optional: supply --no-metrics to avoid writing metrics.json
python embedlab.py search --index ./index --query-dir ./assets/queries --k 5 --no-metrics
```

Advanced:
- `--backbones name1 name2` — if the index contains multi-backbone files (`embeddings.<name>.npy`), the CLI will perform a late-fusion average across backbones for searching.
- `--q-text "some label"` — (optional) zero-shot text queries using CLIP-like encoders. This is guarded by optional imports.

### Analyze
Detect duplicate groups and surface anomalies from an index.

```bash
python embedlab.py analyze --index ./index --dup-threshold 0.92 --anomaly-top 8 --json
```

Options:
- `--k` — k used for neighbor computations and anomaly scoring
- `--ann` / `--n-trees` — use Annoy to accelerate neighbor search for large indexes
- `--no-metrics` — opt out of writing `metrics.json`

## Index format
An index directory contains a few files used by the CLI and tests. The layout is intentionally simple and backward-compatible:

- `embeddings.npy` — (legacy) single-backbone embeddings (float32, shape N x D)
- `embeddings.<name>.npy` — per-backbone embeddings when multiple backbones are saved (multi-backbone support)
- `paths.json` — ordered list of image paths matching embeddings rows
- `meta.json` — metadata (backbone, dim, image_count, seed, etc.)
- `manifest.json` — summary stats written by `embed`
- `metrics.json` — (optional) detailed timings and counters written by embed/search/analyze

The code memory-maps embedding .npy files when possible to avoid loading large arrays into RAM.

## Metrics & observability
When enabled (default), the tool writes `metrics.json` next to the index files. Example fields you can expect:

- `command`: which subcommand emitted the metrics (embed/search/analyze)
- `time_total_sec`, `time_encode_sec`, `images_per_sec` (for embed)
- `ann_build_time`, `ann_query_time_per_query`, `ann_rerank_time_mean_sec` (for tiered/ANN search)

You can opt out with the `--no-metrics` flag on any command.

## Development & tests
Quick start for development:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
pytest -q
```

If you need optional ANN/PCA/CLIP features, install the corresponding packages into the venv (see Requirements above).

## Notes & caveats
- Annoy and some numeric/ML packages may require native compilers (Xcode CLT on macOS). If you hit build errors installing them, ensure you have the platform toolchain available.
- Multi-backbone support uses a simple average late-fusion of normalized embeddings. This is a reasonable default; feel free to experiment with weighted fusion.
- The tiered flow (ANN → exact rerank) provides a good balance of latency and accuracy for large indexes — tune `--candidates` and `--n-trees` for your dataset.

## Contributing / Review checklist
- Run the test suite and linters (`ruff`, `black`) after changes.
- Add focused unit tests when modifying search/analyze code paths (tiered flow, duplicate grouping, anomaly scoring).
- Document any new optional dependencies you add to `requirements-dev.txt` or the README.

