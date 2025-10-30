import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
from embedlib import l2_normalize, save_index, load_index  # noqa: E402


def test_embedding_shape_and_determinism_seeded(tmp_path):
    paths = [f"img_{i}.jpg" for i in range(4)]
    rng = np.random.default_rng(123)
    raw = rng.normal(size=(4, 8)).astype(np.float32)
    normed = l2_normalize(raw)
    save_index(
        tmp_path,
        paths,
        normed,
        {"backbone": "dummy", "dim": 8, "image_count": 4, "seed": 42},
    )
    paths2, emb2, meta2 = load_index(tmp_path)
    save_index(
        tmp_path,
        paths,
        normed,
        {"backbone": "dummy", "dim": 8, "image_count": 4, "seed": 42},
    )
    paths3, emb3, meta3 = load_index(tmp_path)
    assert emb2.shape == (4, 8)
    assert np.allclose(emb2, emb3)
