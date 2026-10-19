import cv2
import numpy as np
from scanner import scan, tiles, render


def test_full_coverage():
    for w, h in [(100, 100), (512, 512), (513, 800), (1400, 1100)]:
        coverage = np.zeros((h, w), dtype=np.uint8)
        for x1, y1, x2, y2 in tiles(w, h, 256, 64):
            coverage[y1:y2, x1:x2] = 1
        assert coverage.all()


def test_synthetic_lines():
    img = np.full((600, 700, 3), 255, dtype=np.uint8)
    cv2.line(img, (50, 100), (650, 100), (0, 0, 0), 5)
    cv2.line(img, (50, 100), (50, 550), (0, 0, 0), 5)
    result = scan(img, tile_size=256, overlap=64)
    assert result["tiles_scanned"] > 1
    assert len(result["line_candidates"]) > 0
    assert render(img, result).shape == img.shape


def test_invalid_overlap():
    import pytest
    with pytest.raises(ValueError):
        list(tiles(100, 100, 128, 128))
