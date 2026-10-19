"""Bayti Vision Scanner QA — experimental, independent of Bayti production.

Deterministic multiscale visual scanning of raster plans.
Requires: numpy, opencv-python. Does NOT claim semantic wall/door recognition.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def tiles(width: int, height: int, size: int = 512, overlap: int = 96):
    """Yield fully covering overlapping image windows in original coordinates."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > overlap >= 0")
    def starts(length):
        if length <= size:
            return [0]
        values = list(range(0, length - size + 1, size - overlap))
        last = length - size
        if values[-1] != last:
            values.append(last)
        return values
    for y in starts(height):
        for x in starts(width):
            yield (x, y, min(x + size, width), min(y + size, height))


def scan(image: np.ndarray, tile_size: int = 512, overlap: int = 96,
         min_line_length: int = 24) -> dict:
    """Scan tiles and propose line segments. Coordinates are image pixels.

    Lines are *candidates*, not confirmed walls. Text, dimensions and furniture
    may be detected too. No 100% accuracy claim is made.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Unsupported image format")
    h, w = gray.shape
    # Local adaptive threshold handles scans with uneven lighting.
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 11)
    seen = set()
    lines = []
    windows = list(tiles(w, h, tile_size, overlap))
    for index, (x1, y1, x2, y2) in enumerate(windows):
        crop = binary[y1:y2, x1:x2]
        raw = cv2.HoughLinesP(crop, 1, np.pi / 180, threshold=35,
                              minLineLength=min_line_length, maxLineGap=8)
        if raw is None:
            continue
        for item in raw[:, 0, :]:
            ax, ay, bx, by = map(int, item)
            p = sorted(((ax + x1, ay + y1), (bx + x1, by + y1)))
            # Coarse quantization only deduplicates near-identical segments.
            key = tuple(round(v / 4) for point in p for v in point)
            if key in seen:
                continue
            seen.add(key)
            lines.append({"start": list(p[0]), "end": list(p[1]),
                          "source_tile": index, "type": "unclassified_line"})
    return {"image_size": {"width": w, "height": h},
            "tile_size": tile_size, "overlap": overlap,
            "tiles_scanned": len(windows), "line_candidates": lines,
            "limitations": "Candidate lines only; no semantic walls, doors, windows or topology yet."}


def render(image: np.ndarray, result: dict) -> np.ndarray:
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[:, :, :3].copy()
    for line in result["line_candidates"]:
        cv2.line(output, tuple(line["start"]), tuple(line["end"]), (0, 180, 0), 2)
    return output


def main():
    parser = argparse.ArgumentParser(description="Bayti isolated experimental raster scanner")
    parser.add_argument("image", type=Path, help="PNG/JPEG floorplan")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=96)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error("Could not read image; provide a valid PNG or JPEG")
    result = scan(image, args.tile_size, args.overlap)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "scan.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    cv2.imwrite(str(args.output / "overlay.png"), render(image, result))
    print(f"Scanned {result['tiles_scanned']} tiles; found {len(result['line_candidates'])} candidate lines")


if __name__ == "__main__":
    main()
