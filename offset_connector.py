"""
P&ID Offset-Page Connector Detector  (Tiled, Connector-Only Edition)
======================================================================
Detects offset page connectors (open pentagon/chevron shapes with labels
like G-30284) on large P&ID engineering drawings.

This is a connector-only build. Flow arrow detection was removed: on real
P&ID exports, arrowheads are either thin open-stroke chevrons or filled
triangles fused directly to their pipe line with no rendering gap, and they
are visually near-identical to valve symbols (which are two triangles
meeting at a point). A reliable arrow detector needs line-convergence
analysis (Hough transform) or template matching, not simple contour-blob
detection — that's a separate follow-up.

Works on full-resolution P&ID images (3000+ px) by slicing into overlapping
tiles, detecting within each tile, converting coordinates back to full-image
space, then merging duplicate detections with NMS.

Usage:
    python pid_connector_detector.py -i page_75.png
    python pid_connector_detector.py -i page_75.png -o results/ --tile-size 768 --overlap 100
    python pid_connector_detector.py -i page_75.png --no-ocr --debug
    python pid_connector_detector.py -i page_75.png --show-tiles
"""

import cv2
import numpy as np
import json
import argparse
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    def as_xyxy(self):
        return self.x, self.y, self.x + self.w, self.y + self.h

    def area(self):
        return self.w * self.h


@dataclass
class Detection:
    type: str                     # always "connector" in this build
    direction: Optional[str]      # "left" | "right"
    bbox: BBox
    label_region: Optional[BBox]  # region to run OCR over
    ocr_text: Optional[str]
    confidence: float


# ──────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────

def load_and_binarize(path: str) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=4
    )
    return img, binary


# ──────────────────────────────────────────────
# Tiling engine
# ──────────────────────────────────────────────

def make_tiles(img_h: int, img_w: int,
               tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """
    Return list of (x1, y1, x2, y2) tile regions covering the full image.
    Tiles overlap so symbols on boundaries aren't missed. The last tile in
    each row/col is extended to reach the image edge.
    """
    stride = tile_size - overlap
    tiles = []
    y = 0
    while y < img_h:
        x = 0
        y2 = min(y + tile_size, img_h)
        while x < img_w:
            x2 = min(x + tile_size, img_w)
            tiles.append((x, y, x2, y2))
            if x2 == img_w:
                break
            x += stride
        if y2 == img_h:
            break
        y += stride
    return tiles


# ──────────────────────────────────────────────
# Connector detection (chevron / pentagon)
# ──────────────────────────────────────────────

def is_chevron_shape(approx: np.ndarray,
                     bbox_w: int, bbox_h: int) -> tuple[bool, Optional[str]]:
    """
    Determine if a polygon looks like a connector chevron and which way it points.

    A real connector is a simple pentagon: one flat (open) end with two
    corners, two roughly-parallel horizontal edges (top/bottom), and one
    pointed tip. 3-6 vertices after simplification, exactly one pointed end.
    """
    n = len(approx)
    if not (3 <= n <= 6):
        return False, None

    aspect = bbox_w / max(bbox_h, 1)
    if not (1.5 <= aspect <= 9.0):
        return False, None

    pts = approx.reshape(-1, 2).astype(float)
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    span = x_max - x_min
    if span < 1:
        return False, None

    # Find the single most extreme point on each side, then check whether
    # any other point is close to it in x (within 12% of span). If the
    # extreme point stands alone, that side is the tip; if it has neighbors,
    # that side is the flat/open end.
    near_zone = span * 0.12
    left_neighbors  = pts[pts[:, 0] <= x_min + near_zone]
    right_neighbors = pts[pts[:, 0] >= x_max - near_zone]
    n_left, n_right = len(left_neighbors), len(right_neighbors)

    if n_right == 1 and n_left >= 2:
        return True, "right"
    if n_left == 1 and n_right >= 2:
        return True, "left"

    # Both extremes isolated — connector likely clipped at a tile/image
    # boundary, cutting off the flat end's second corner. x_min≈0 means the
    # open end was clipped on the left, so the tip is on the right.
    if n_right == 1 and n_left == 1:
        return (True, "right") if x_min <= 1.0 else (True, "left")

    return False, None


def shape_purity_ok(cnt: np.ndarray, bw: int, bh: int) -> bool:
    """
    Reject contours from merged/touching line-work rather than a single
    clean chevron outline.

    Measured on real connectors (full-res P&ID export and clean screenshot
    crops): extent 0.35-0.61, solidity 0.51-0.76. Band is set wide enough to
    cover both while still rejecting clearly-merged clutter — very low
    extent means the bbox spans mostly empty space (merged distant
    elements); very high extent/solidity means a filled/dense blob.
    """
    area = cv2.contourArea(cnt)
    bbox_area = bw * bh
    if bbox_area <= 0:
        return False

    extent = area / bbox_area
    if not (0.25 <= extent <= 0.85):
        return False

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(hull_area, 1)
    if not (0.40 <= solidity <= 0.95):
        return False

    # Convexity defects: a clean chevron has few significant concave
    # points. Merged clutter (e.g. instrument clusters, dashed boxes) has
    # many.
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is not None and len(hull_idx) > 3:
        try:
            defects = cv2.convexityDefects(cnt, hull_idx)
        except cv2.error:
            defects = None
        if defects is not None:
            diag = (bw ** 2 + bh ** 2) ** 0.5
            sig_defects = sum(1 for d in defects if (d[0][3] / 256.0) > diag * 0.03)
            if sig_defects > 4:
                return False

    return True


def detect_connectors_in_tile(tile_img: np.ndarray,
                               tile_bin: np.ndarray,
                               cfg: dict,
                               debug: bool = False) -> list[Detection]:
    """Run connector detection on a single tile."""
    detections = []
    # No morphological closing: it merges nearby unrelated line-work in
    # dense P&ID regions. Connectors are already closed outlines.
    contours, _ = cv2.findContours(tile_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg["conn_min_area"] or area > cfg["conn_max_area"]:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < bh * 1.5:
            continue

        if not shape_purity_ok(cnt, bw, bh):
            if debug:
                print(f"    REJECTED (purity) area={area:.0f} bbox=({bw}x{bh})")
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        is_chev, direction = is_chevron_shape(approx, bw, bh)

        if debug:
            print(f"    area={area:.0f} pts={len(approx)} "
                  f"asp={bw/max(bh,1):.2f} chev={is_chev} dir={direction}")

        if not is_chev:
            continue

        pad_x = int(bw * 0.12)
        pad_y = int(bh * 0.15)
        label_region = BBox(x + pad_x, y + pad_y,
                            bw - 2 * pad_x, bh - 2 * pad_y)

        detections.append(Detection(
            type="connector",
            direction=direction,
            bbox=BBox(x, y, bw, bh),
            label_region=label_region,
            ocr_text=None,
            confidence=0.85
        ))

    return detections


# ──────────────────────────────────────────────
# Detection thresholds
# ──────────────────────────────────────────────

def make_config() -> dict:
    """
    Absolute pixel-area thresholds for connector detection.

    Connector chevrons are drawn at a fixed drafting symbol size and do NOT
    scale with overall page export resolution — measured directly from a
    real 3024×2160 P&ID export, a connector is ~150-300px wide and
    ~25-60px tall (area roughly 1,700-3,400 px²). Margin is added on both
    sides to cover other export resolutions and minor symbol-size variation
    between drawings.
    """
    return {
        "conn_min_area": 600,
        "conn_max_area": 25_000,
    }


# ──────────────────────────────────────────────
# NMS
# ──────────────────────────────────────────────

def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a.as_xyxy()
    bx1, by1, bx2, by2 = b.as_xyxy()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.area() + b.area() - inter
    return inter / max(union, 1)


def nms(detections: list[Detection], iou_thresh: float = 0.35) -> list[Detection]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept = []
    for det in detections:
        if not any(iou(k.bbox, det.bbox) > iou_thresh for k in kept):
            kept.append(det)
    return kept


# ──────────────────────────────────────────────
# OCR
# ──────────────────────────────────────────────

def run_ocr(img: np.ndarray, detections: list[Detection]) -> None:
    connectors = [d for d in detections if d.label_region]
    if not connectors:
        return
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    except ImportError:
        print("  [OCR] easyocr not installed — skipping")
        return

    ih, iw = img.shape[:2]
    for det in connectors:
        lr = det.label_region
        x1 = max(0, lr.x);      y1 = max(0, lr.y)
        x2 = min(iw, lr.x + lr.w); y2 = min(ih, lr.y + lr.h)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        scale = max(1, int(80 / max(crop.shape[:2])))
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        results = reader.readtext(crop, detail=0, paragraph=True)
        det.ocr_text = " ".join(results).strip() if results else None


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

CONN_COLOR = (0, 200, 0)


def draw_detections(img: np.ndarray, detections: list[Detection],
                    tile_grid: Optional[list] = None) -> np.ndarray:
    vis = img.copy()
    ih, iw = vis.shape[:2]

    if tile_grid:
        for (tx1, ty1, tx2, ty2) in tile_grid:
            cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), (200, 200, 200), 1)

    # Fixed small font — independent of image resolution so labels stay tiny.
    # Connectors on a 3000px-wide P&ID are ~200×40px; a 0.35 font scale
    # produces ~10px-tall text which is legible without covering the symbol.
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.35
    FONT_THICK = 1
    BOX_THICK  = max(1, iw // 2000)   # thin bounding-box outline
    PAD        = 2                     # pixels of padding around text bg

    for det in detections:
        x, y, bw, bh = det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h
        x2, y2 = x + bw, y + bh

        # ── bounding box outline (thin) ──────────────────────────────────
        cv2.rectangle(vis, (x, y), (x2, y2), CONN_COLOR, BOX_THICK)

        # ── label text ──────────────────────────────────────────────────
        # Use full words "left" / "right" so direction is unambiguous
        dir_word = det.direction if det.direction else "unknown"
        # e.g. "conn(right) G-30284"  or just "conn(left)"
        label = f"conn({dir_word})"
        if det.ocr_text:
            label += f" {det.ocr_text}"

        (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
        label_h = th + bl + 2 * PAD

        # Place label ABOVE the bounding box if room, else BELOW
        if y >= label_h + 2:
            # above: text baseline sits just above the top edge of the box
            bg_y1 = y - label_h - 1
            bg_y2 = y - 1
            txt_y = bg_y2 - PAD - bl
        else:
            # below: label goes under the bottom edge
            bg_y1 = y2 + 1
            bg_y2 = y2 + label_h + 1
            txt_y = bg_y1 + PAD + th

        # Clamp x so label doesn't run off the right edge
        txt_x = min(x, iw - tw - PAD - 1)
        bg_x1 = txt_x - PAD
        bg_x2 = txt_x + tw + PAD

        # Small filled rectangle behind the text only (not over the connector)
        cv2.rectangle(vis, (bg_x1, bg_y1), (bg_x2, bg_y2), CONN_COLOR, -1)
        cv2.putText(vis, label, (txt_x, txt_y),
                    FONT, FONT_SCALE, (255, 255, 255), FONT_THICK, cv2.LINE_AA)

    return vis


# ──────────────────────────────────────────────
# JSON export
# ──────────────────────────────────────────────

def to_json(detections: list[Detection], image_path: str) -> dict:
    return {
        "image": image_path,
        "total_connectors": len(detections),
        "detections": [
            {
                "id": i,
                "type": "connector",
                "direction": d.direction,
                "bbox": {"x": d.bbox.x, "y": d.bbox.y,
                         "w": d.bbox.w, "h": d.bbox.h},
                "label_region": (
                    {"x": d.label_region.x, "y": d.label_region.y,
                     "w": d.label_region.w, "h": d.label_region.h}
                    if d.label_region else None
                ),
                "ocr_text": d.ocr_text,
                "confidence": d.confidence,
            }
            for i, d in enumerate(detections)
        ]
    }


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def run(input_path: str,
        output_dir: str,
        tile_size: int  = 512,
        overlap:   int  = 80,
        use_ocr:   bool = True,
        debug:     bool = False,
        show_tiles: bool = False):

    print(f"\n{'='*55}")
    print(f"  P&ID Connector Detector  (tiled)")
    print(f"  Input     : {input_path}")
    print(f"  Output    : {output_dir}")
    print(f"  Tile size : {tile_size}  Overlap: {overlap}")
    print(f"{'='*55}")

    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/5] Loading image...")
    img, binary = load_and_binarize(input_path)
    ih, iw = img.shape[:2]
    print(f"      Size: {iw} × {ih} px")

    cfg = make_config()
    print(f"      Connector area range : {cfg['conn_min_area']:,} – {cfg['conn_max_area']:,} px²")

    tiles = make_tiles(ih, iw, tile_size, overlap)
    print(f"\n[2/5] Tiling: {len(tiles)} tiles ({tile_size}px, overlap={overlap}px)")

    all_connectors: list[Detection] = []
    for idx, (tx1, ty1, tx2, ty2) in enumerate(tiles):
        tile_img = img[ty1:ty2, tx1:tx2]
        tile_bin = binary[ty1:ty2, tx1:tx2]

        if debug:
            print(f"\n  Tile {idx+1}/{len(tiles)}  ({tx1},{ty1})→({tx2},{ty2})")

        conns = detect_connectors_in_tile(tile_img, tile_bin, cfg, debug)
        for d in conns:
            d.bbox.x += tx1
            d.bbox.y += ty1
            if d.label_region:
                d.label_region.x += tx1
                d.label_region.y += ty1
        all_connectors.extend(conns)

    print(f"      Raw detections: {len(all_connectors)} (before NMS)")

    print("\n[3/5] Merging detections (NMS)...")
    connectors = nms(all_connectors)
    print(f"      After NMS: {len(connectors)} connectors")

    if use_ocr and connectors:
        print("\n[4/5] Running OCR on connector labels...")
        run_ocr(img, connectors)
        for d in connectors:
            if d.ocr_text:
                print(f"      {str(d.direction):>5}  →  '{d.ocr_text}'")
    else:
        print("\n[4/5] OCR skipped.")

    print("\n[5/5] Saving outputs...")
    stem = Path(input_path).stem

    vis = draw_detections(img, connectors, tile_grid=tiles if show_tiles else None)
    vis_path  = os.path.join(output_dir, f"{stem}_connectors.png")
    json_path = os.path.join(output_dir, f"{stem}_connectors.json")

    cv2.imwrite(vis_path, vis)
    result = to_json(connectors, input_path)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"      Visualization → {vis_path}")
    print(f"      JSON          → {json_path}")
    print(f"\n  ✓ Done  |  {result['total_connectors']} connectors found\n")

    return vis_path, json_path, result


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P&ID tiled connector detector")
    parser.add_argument("--input",      "-i", required=True)
    parser.add_argument("--output",     "-o", default="results")
    parser.add_argument("--tile-size",  "-t", type=int, default=512)
    parser.add_argument("--overlap",         type=int, default=80)
    parser.add_argument("--no-ocr",     action="store_true")
    parser.add_argument("--debug",      action="store_true")
    parser.add_argument("--show-tiles", action="store_true")
    args = parser.parse_args()

    run(
        input_path  = args.input,
        output_dir  = args.output,
        tile_size   = args.tile_size,
        overlap     = args.overlap,
        use_ocr     = not args.no_ocr,
        debug       = args.debug,
        show_tiles  = args.show_tiles,
    )