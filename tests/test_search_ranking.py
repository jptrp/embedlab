import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
from embedlib import save_index, cosine_similarity_matrix  # noqa: E402


def test_search_ranking_stability_tiny(tmp_path):
    paths = ["a.jpg", "b.jpg", "c.jpg"]
    E = np.array([[1, 0, 0], [0.8, 0.2, 0], [0, 1, 0]], dtype=np.float32)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    save_index(
        tmp_path,
        paths,
        E,
        {"backbone": "dummy", "dim": 3, "image_count": 3, "seed": 42},
    )
    q = np.array([[0.9, 0.1, 0]], dtype=np.float32)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    S = cosine_similarity_matrix(q, E)[0]
    ordering = list(np.argsort(-S))
    assert ordering == [0, 1, 2]
