## linefun2.py
## Line + Arrow detection for P&ID diagrams.
## v2 – reduced false positives via:
##   1. Narrow / aspect-correct arrow templates (3:1 ratio, not square)
##   2. Aspect-ratio gate on matched region (rejects circles, square blobs)
##   3. Tighter fill-ratio window (0.12 – 0.48)
##   4. Improved valve rejection: convex-hull solidity + centroid opposition
##   5. Stricter snap_radius (arrow_size * 2, was *3)
##   6. Minimum match score per-template raised at call site

import os, math
import fitz
import cv2
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import Counter


# ── Page extraction ────────────────────────────────────────────────────────────

def extract_single_page_image(pdf_path, output_image_path, page_number, dpi=300):
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    os.makedirs(output_image_path, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        if not (1 <= page_number <= doc.page_count):
            raise ValueError(f"page {page_number} out of range")
        pix = doc.load_page(page_number - 1).get_pixmap(
            matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
        out = os.path.join(output_image_path, f"page_{page_number}.png")
        pix.save(out)
    finally:
        doc.close()
    return os.path.abspath(out)


# ── Morphological line detection ───────────────────────────────────────────────

def extract_lines_without_ocr(image_path, dotted_kernel, solid_kernel):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    del img

    def _lines(src, k):
        return cv2.add(
            cv2.morphologyEx(src, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)), iterations=2),
            cv2.morphologyEx(src, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)), iterations=2))

    solid  = _lines(binary, solid_kernel)
    dotted = _lines(binary, dotted_kernel)
    del binary
    return dotted, solid, cv2.subtract(dotted, solid)


def save_morphology_results(page_number, lines_solid, lines_dotted, lines_diff, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved = {}
    for label, arr in [("solid", lines_solid), ("dotted", lines_dotted), ("diff", lines_diff)]:
        path = os.path.join(output_dir, f"lines_morphology_{label}_pn_{page_number}.png")
        cv2.imwrite(path, arr)
        saved[label] = path
        print(f"  [{label:>6}] saved → {path}")
    return saved


# ── Hough line detection ───────────────────────────────────────────────────────

def detect_hough_lines(
    lines_diff,
    lines_only_solid,
    image_path,
    hough_transform_output_path,
    page_number,
    DPI,
    config,
):
    import os
    import cv2
    import json
    import math
    import numpy as np

    cfg = config

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    def _hough(mask):
        edges = cv2.Canny(
            mask,
            cfg.canny_low,
            cfg.canny_high,
            apertureSize=3,
        )

        return cv2.HoughLinesP(
            edges,
            cfg.rho,
            cfg.theta,
            cfg.threshold,
            minLineLength=cfg.min_line_length,
            maxLineGap=cfg.max_line_gap,
        )

    # ------------------------------------------------------------------
    # Detect Hough lines
    # ------------------------------------------------------------------

    lines_dotted_total = _hough(lines_diff)
    lines_solid_total = _hough(lines_only_solid)

    print(
        f"  [solid ] {len(lines_solid_total) if lines_solid_total is not None else 0} segment(s)"
    )
    print(
        f"  [dotted] {len(lines_dotted_total) if lines_dotted_total is not None else 0} segment(s)"
    )

    # ------------------------------------------------------------------
    # Visualization image
    # ------------------------------------------------------------------

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------

    line_json = []
    line_id = 1

    line_sets = [
        (lines_dotted_total, (0, 180, 0), "dotted"),
        (lines_solid_total, (180, 0, 0), "solid"),
    ]

    for segs, color, line_type in line_sets:

        if segs is None:
            continue

        for s in segs:

            s = np.asarray(s)

            # Supports both OpenCV 4 and OpenCV 5 outputs
            if s.ndim == 2:
                x1, y1, x2, y2 = s[0]
            else:
                x1, y1, x2, y2 = s

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # Draw line
            cv2.line(
                vis,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            # Geometry
            length = math.hypot(x2 - x1, y2 - y1)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

            line_json.append(
                {
                    "id": line_id,
                    "line_type": line_type,
                    "start": {
                        "x": x1,
                        "y": y1,
                    },
                    "end": {
                        "x": x2,
                        "y": y2,
                    },
                    "center": {
                        "x": round((x1 + x2) / 2, 2),
                        "y": round((y1 + y2) / 2, 2),
                    },
                    "bbox": {
                        "xmin": min(x1, x2),
                        "ymin": min(y1, y2),
                        "xmax": max(x1, x2),
                        "ymax": max(y1, y2),
                    },
                    "length": round(length, 2),
                    "angle": round(angle, 2),
                }
            )

            line_id += 1

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    os.makedirs(hough_transform_output_path, exist_ok=True)

    image_output = os.path.join(
        hough_transform_output_path,
        f"line_hough_transform_{page_number}.png",
    )

    json_output = os.path.join(
        hough_transform_output_path,
        f"line_hough_transform_{page_number}.json",
    )

    cv2.imwrite(image_output, vis)

    with open(json_output, "w") as f:
        json.dump(line_json, f, indent=4)

    print(f"  [image ] saved → {image_output}")
    print(f"  [json  ] saved → {json_output}")
    print(f"  Total detected lines : {len(line_json)}")

    return lines_solid_total, lines_dotted_total