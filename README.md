# Bayti Vision Scanner — QA experiment

**Completely independent experiment.** This repository is NOT the Bayti production repository. No deployment, paid APIs, or modifications to the original Bayti repository.

## What works in this initial prototype

- Scan PNG/JPEG plans with **overlapping windows**, retaining absolute pixel coordinates.
- Local adaptive thresholding and per-window line proposals (OpenCV Hough transform).
- JSON output containing candidate line endpoints and tile provenance.
- Green overlay showing proposed lines for human inspection.
- Synthetic regression tests for complete tile coverage and basic detection.

## What does NOT work yet

This is a **visual scanning baseline**, not an intelligent floor-plan reader. It does not reliably classify walls vs dimension lines, detect doors/windows, handle curves as curves, infer rooms, or guarantee 100% detection. The Hough line proposals can fragment long walls and produce false positives. There are no trained vision models or QA benchmark results yet.

## Run locally

```bash
python -m pip install -r requirements.txt
python scanner.py path/to/plan.png --output output
python -m pytest -q
```

Outputs: `output/scan.json` and `output/overlay.png`.

## Next QA gates

1. Evaluate on diverse, separately annotated floor plans (not only synthetic examples).
2. Add line tracing/merging across tile boundaries and curve handling.
3. Classify wall / annotation / furniture using evidence from the full plan.
4. Detect openings and associate each with a host wall.
5. Compare precision, recall, and topology against Bayti *without editing Bayti*.
6. Keep holdout plans untouched until final evaluation.

**Never claim 100% accuracy without external holdout evidence.**
