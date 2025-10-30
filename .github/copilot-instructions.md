## Quick orientation

- Repo purpose: a small CLI that embeds images, performs nearest-neighbor search, and analyzes duplicates/anomalies using image embeddings.
- Main entry: `embedlab.py` (calls `src/embedlab_cli.main`). See `README.md` for example CLI invocations:
  - `python embedlab.py embed --images-dir ./assets/images --out ./index`
  - `python embedlab.py search --index ./index --query-dir ./assets/queries --k 5 --json`

## Key modules & data formats (read before editing behavior)
- `src/embedlib.py` — core logic for listing images, backbone encoding, normalization, similarity, index save/load, search, and analysis. Important functions:
  - `op_embed(images_dir, out_dir, seed=42)` — runs encoding flow and writes index.
  - `save_index(index_dir, paths, embeddings, meta)` / `load_index(index_dir)` — index is saved as `embeddings.npy`, `paths.json`, and `meta.json`.
  - `l2_normalize()` — embeddings are normalized on save; cosine similarity assumes normalized vectors (dot product used).
  - `analyze_index(index_dir, dup_threshold, anomaly_top, knn=5)` — produces `duplicate_groups` and `anomalies` using an internal union-find and mean neighbor isolation score.

## Patterns & conventions to follow
- Embeddings: always stored as float32 2D arrays in `embeddings.npy`. `meta.json` includes keys: `backbone`, `dim`, `image_count`, `created_at`, `seed` (see how `op_embed` constructs meta).
- Determinism: call `set_seed(seed)` (defined in `embedlib`) when reproducibility matters — tests rely on deterministic behavior.
- Supported image types: extensions in `IMG_EXTS` (see `embedlib.list_images`) — add extensions here for new image formats.
- Backbone: `TorchResNet18Backbone` is a thin wrapper that uses torchvision weights and returns raw features (fc -> Identity). Default device is CPU in this repo.

## Testing & developer workflow
- Tests: run `pytest -q` from repository root. `pyproject.toml` sets `pythonpath = ["src"]` so tests import `embedlib` directly.
- Quick checks used in CI/tests:
  - saving/loading index: `save_index(tmp_path, paths, embeddings, meta)` and `load_index` are used heavily in tests.
  - similarity expectations: `cosine_similarity_matrix` returns dot-products of normalized vectors; tests assert ordering/thresholds based on that.

## Examples to cite in PRs or fixes
- Duplicate threshold boundary: `tests/test_duplicates_boundary.py` demonstrates how `analyze_index(..., dup_threshold=0.95, ...)` groups images when cosine >= threshold.
- Determinism example: `tests/test_embed_determinism.py` shows writing and reloading the same embeddings to assert equality.

## Integration notes & gotchas
- Index layout must not change silently: other tooling and tests expect three files in the index directory (`embeddings.npy`, `paths.json`, `meta.json`). If you need to change layout, update tests and README.
- Normalization: code assumes embeddings are normalized. If modifying encoding/backbone, preserve normalization or update search/analyze accordingly.
- Performance: backbone currently runs on CPU; if enabling GPU, make device configurable and ensure tests run on CI without GPU by keeping CPU default.

## When editing, please include small examples and tests
- For any behavioral change, add a focused unit test under `tests/` (see existing tests for style). Prefer small, deterministic numpy arrays rather than heavy model runs to keep tests fast.

---
If any of this is unclear or you want more specifics (example patch for a refactor, CI instructions, or a sample test), tell me which area to expand and I will iterate.
