from __future__ import annotations
import os
import json
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def list_images(dir_path: str) -> List[str]:
    out = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            ext = os.path.splitext(f.lower())[1]
            if ext in IMG_EXTS:
                out.append(os.path.join(root, f))
    out.sort()
    return out


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B.T  # assuming normalized


@dataclass
class BackboneInfo:
    name: str
    dim: int


class TorchResNet18Backbone:
    def __init__(self, device: str = "cpu"):
        import torch
        from torchvision.models import resnet18, ResNet18_Weights

        self.device = torch.device(device)
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval().to(self.device)
        self.model = model
        self.preprocess = weights.transforms()
        self.dim = 512

    def encode_paths(self, paths: List[str], batch_size: int = 1) -> np.ndarray:
        """Encode image files into a (N, dim) numpy float32 array.

        Processes images in batches (batch_size) to improve throughput.
        """
        import torch
        from PIL import Image

        feats: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i : i + batch_size]
                batch_tensors = []
                for p in batch_paths:
                    img = Image.open(p).convert("RGB")
                    x = self.preprocess(img)
                    batch_tensors.append(x)
                if not batch_tensors:
                    continue
                x = torch.stack(batch_tensors).to(self.device)
                y = self.model(x).cpu().numpy()
                for row in y:
                    feats.append(row)
        return np.array(feats, dtype=np.float32)

    def info(self) -> BackboneInfo:
        return BackboneInfo(name="torchvision.resnet18_imagenet1k", dim=self.dim)


def save_index(
    index_dir: str, paths: List[str], embeddings: np.ndarray, meta: Dict[str, Any]
):
    os.makedirs(index_dir, exist_ok=True)
    np.save(os.path.join(index_dir, "embeddings.npy"), embeddings.astype(np.float32))
    with open(os.path.join(index_dir, "paths.json"), "w") as f:
        json.dump(paths, f)
    with open(os.path.join(index_dir, "meta.json"), "w") as f:
        json.dump(meta, f)


def load_index(index_dir: str) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    paths = json.load(open(os.path.join(index_dir, "paths.json"), "r"))
    # memory-map the embeddings to avoid loading the entire array when possible
    embeddings = np.load(os.path.join(index_dir, "embeddings.npy"), mmap_mode="r")
    meta = json.load(open(os.path.join(index_dir, "meta.json"), "r"))
    return paths, embeddings, meta


def op_embed(images_dir: str, out_dir: str, seed: int = 42, batch_size: int = 1):
    """Embed all images under `images_dir` and save an index to `out_dir`.

    Adds reproducibility via `seed`, supports batching for throughput, and writes
    a short `manifest.json` with runtime metrics.
    """
    set_seed(seed)
    t0 = time.time()
    paths = list_images(images_dir)
    if not paths:
        raise SystemExit(f"No images found under: {images_dir}")
    backbone = TorchResNet18Backbone(device="cpu")
    raw = backbone.encode_paths(paths, batch_size=batch_size)
    normed = l2_normalize(raw)
    meta = {
        "backbone": backbone.info().name,
        "dim": int(normed.shape[1]),
        "image_count": len(paths),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "batch_size": int(batch_size),
    }
    save_index(out_dir, paths, normed, meta)
    dt = time.time() - t0
    ips = len(paths) / dt if dt > 0 else float("inf")
    manifest = {
        "images": len(paths),
        "dim": int(normed.shape[1]),
        "time_sec": dt,
        "images_per_sec": ips,
        "backbone": meta["backbone"],
        "seed": seed,
        "batch_size": int(batch_size),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as mf:
        json.dump(manifest, mf)
    print(
        f"[embed] images={len(paths)} dim={normed.shape[1]} time_sec={dt:.2f} images/sec={ips:.2f} backbone={meta['backbone']}"
    )


def topk_search(
    index_dir: str,
    query_dir: str,
    k: int,
    batch_size: int = 1,
    seed: int = 42,
    ann: bool = False,
    n_trees: int = 10,
) -> Any:
    """Search nearest neighbors for query images.

    Returns a list of dicts with keys: `query`, `results` where each result includes
    `path`, `score`, `distance` (1-score) and `reason`. Each result dict is
    augmented with `backbone` and `dim` from the index meta for explainability.
    """
    paths, embeddings, meta = load_index(index_dir)
    qpaths = list_images(query_dir)
    if not qpaths:
        raise SystemExit(f"No query images found under: {query_dir}")
    set_seed(seed)
    backbone = TorchResNet18Backbone(device="cpu")
    qemb = l2_normalize(backbone.encode_paths(qpaths, batch_size=batch_size))
    # Exact similarity matrix
    S = cosine_similarity_matrix(qemb, embeddings)
    ann_times = None
    if ann:
        try:
            from annoy import AnnoyIndex

            dim = int(qemb.shape[1])
            t0 = time.time()
            aidx = AnnoyIndex(dim, "angular")
            # embeddings may be mmap; iterate rows
            for ii in range(int(embeddings.shape[0])):
                aidx.add_item(ii, embeddings[ii].astype(np.float32))
            aidx.build(n_trees)
            ann_build_time = time.time() - t0
            # measure ann query time
            t0 = time.time()
            for qq in range(len(qemb)):
                _ = aidx.get_nns_by_vector(qemb[qq].astype(np.float32), k)
            ann_times = (time.time() - t0) / max(1, len(qemb))
        except Exception:
            ann = False
            ann_build_time = None
    results = []
    for qi, qpath in enumerate(qpaths):
        scores = S[qi]
        top_idx = np.argsort(-scores)[:k]
        items = []
        for j in top_idx:
            sc = float(round(float(scores[j]), 6))
            items.append(
                {
                    "path": paths[j],
                    "score": sc,
                    "distance": float(round(1.0 - sc, 6)),
                    "reason": "cosine_similarity",
                }
            )
        results.append(
            {
                "query": qpath,
                "results": items,
                "backbone": meta.get("backbone"),
                "dim": meta.get("dim"),
            }
        )
    # attach timings if ann was requested
    if ann:
        return {
            "results": results,
            "ann_build_time": ann_build_time,
            "ann_query_time_per_query": ann_times,
        }
    return results


def analyze_index(
    index_dir: str,
    dup_threshold: float,
    anomaly_top: int,
    knn: int = 5,
    ann: bool = False,
    n_trees: int = 10,
):
    paths, embeddings, meta = load_index(index_dir)
    N = int(embeddings.shape[0])
    if N == 0:
        return {"duplicate_groups": [], "anomalies": []}
    # full similarity matrix (not filled) for scoring; use a copy for
    # thresholding
    S_full = cosine_similarity_matrix(embeddings, embeddings)
    S = S_full.copy()
    np.fill_diagonal(S, -np.inf)
    parent = list(range(N))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Build k-NN graph and union nodes where similarity >= threshold.
    if ann:
        try:
            from annoy import AnnoyIndex

            dim = int(embeddings.shape[1])
            aidx = AnnoyIndex(dim, "angular")
            for ii in range(N):
                aidx.add_item(ii, embeddings[ii].astype(np.float32))
            aidx.build(n_trees)
            for i in range(N):
                nbrs = aidx.get_nns_by_item(i, knn + 1)
                # drop self if present
                nbrs = [j for j in nbrs if j != i][:knn]
                for j in nbrs:
                    if S_full[i, j] + 1e-6 >= dup_threshold:
                        union(i, j)
        except Exception:
            # fallback to exact
            for i in range(N):
                for j in range(i + 1, N):
                    if S[i, j] + 1e-6 >= dup_threshold:
                        union(i, j)
    else:
        for i in range(N):
            # get top-k neighbors for i (excluding diagonal)
            cos = S[i].copy()
            cos[cos == -np.inf] = -1.0
            idx = np.argsort(-cos)[:knn]
            for j in idx:
                if S_full[i, j] + 1e-6 >= dup_threshold:
                    union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(N):
        r = find(i)
        groups.setdefault(r, []).append(i)

    # preserve legacy simple grouping (list of list of paths)
    dup_groups = [
        [paths[i] for i in idxs] for idxs in groups.values() if len(idxs) >= 2
    ]

    # richer explainability (detailed groups with scores)
    dup_groups_detailed = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        rep = idxs[0]
        members = []
        for i in idxs:
            members.append(
                {
                    "path": paths[i],
                    "score_to_rep": float(round(float(S_full[rep, i]), 6)),
                }
            )
        dup_groups_detailed.append(
            {"members": members, "reason": f"cosine>={dup_threshold}"}
        )

    K = min(knn, N - 1) if N > 1 else 0
    anomalies: List[str] = []
    anomalies_detailed = []
    if K > 0:
        isolation: List[Tuple[int, float]] = []
        for i in range(N):
            cos = S[i].copy()
            cos[cos == -np.inf] = -1.0
            idx = np.argsort(-cos)[:K]
            dists = 1.0 - cos[idx]
            isolation.append((i, float(np.mean(dists))))
        isolation.sort(key=lambda t: t[1], reverse=True)
        anomalies = [paths[i] for (i, _) in isolation[:anomaly_top]]
        anomalies_detailed = [
            {"path": paths[i], "isolation_score": float(round(score, 6))}
            for (i, score) in isolation[:anomaly_top]
        ]

    return {
        "duplicate_groups": dup_groups,
        "anomalies": anomalies,
        "meta": {"backbone": meta.get("backbone"), "dim": meta.get("dim")},
        "duplicate_groups_detailed": dup_groups_detailed,
        "anomalies_detailed": anomalies_detailed,
    }
