from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from hsi_plotter.io.curve_data_reader import CurveDataError, read_curve_mat


def test_read_curve_mat_accepts_legacy_matrices(tmp_path: Path) -> None:
    path = tmp_path / "MHT.mat"
    precision = np.vstack([np.linspace(0, 1, 51), np.linspace(0.1, 1, 51)])
    success = np.vstack([np.linspace(1, 0, 51), np.linspace(0.9, 0, 51)])
    savemat(path, {"distance_rec": precision, "PASCAL_rec": success})

    data = read_curve_mat(path)

    assert data.display_name == "MHT"
    assert np.allclose(data.precision_curve, np.mean(precision, axis=0))
    assert np.allclose(data.success_curve, np.mean(success, axis=0))
    assert data.dp == pytest.approx(data.precision_curve[20])
    assert data.auc == pytest.approx(np.mean(data.success_curve[1:]))


def test_read_curve_mat_prefers_exported_mean_curves(tmp_path: Path) -> None:
    path = tmp_path / "paper.mat"
    precision = np.linspace(0, 1, 51)
    success = np.linspace(1, 0, 51)
    savemat(
        path,
        {
            "distance_rec": np.zeros((2, 51)),
            "PASCAL_rec": np.zeros((2, 51)),
            "distance_rec_mean": precision,
            "PASCAL_rec_mean": success,
            "DP": 0.123,
            "AUC": 0.456,
        },
    )

    data = read_curve_mat(path)

    assert np.allclose(data.precision_curve, precision)
    assert data.dp == pytest.approx(0.123)
    assert data.auc == pytest.approx(0.456)


def test_read_curve_mat_rejects_missing_or_wrong_length_curves(tmp_path: Path) -> None:
    path = tmp_path / "broken.mat"
    savemat(path, {"distance_rec": np.zeros((1, 50))})

    with pytest.raises(CurveDataError, match="51"):
        read_curve_mat(path)
