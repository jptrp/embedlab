import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
from embedlib import save_index, analyze_index, l2_normalize  # noqa: E402


def test_duplicate_grouping_at_threshold_boundary(tmp_path):
    v1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    v2 = np.array([[0.95, (1 - 0.95**2) ** 0.5, 0.0]], dtype=np.float32)
    E = l2_normalize(np.vstack([v1, v2]))
    paths = ["x.jpg", "y.jpg"]
    save_index(
        tmp_path,
        paths,
        E,
        {"backbone": "dummy", "dim": 3, "image_count": 2, "seed": 42},
    )
    out = analyze_index(tmp_path, dup_threshold=0.95, anomaly_top=1, knn=1)
    groups = out["duplicate_groups"]
    assert len(groups) == 1 and set(groups[0]) == set(paths)
