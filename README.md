# embedlab — Image Embeddings, Search, Dedupe & Anomaly Detection (CLI)

CLI tool that:
- **Embeds** images into fixed-length vectors using a reasonable vision backbone (default: torchvision ResNet18/IMAGENET1K).
- **Searches** nearest neighbors for query images with cosine similarity.
- **Analyzes** near-duplicates and anomalies using only embeddings.

## Requirements
- Python 3.10+
- Runtime deps for `embed`: `torch`, `torchvision`, `Pillow`, `numpy`

## CLI
### Embed
`python embedlab.py embed --images-dir ./assets/images --out ./index`

### Search
`python embedlab.py search --index ./index --query-dir ./assets/queries --k 5 --json`

### Analyze
`python embedlab.py analyze --index ./index --dup-threshold 0.92 --anomaly-top 8 --json`
