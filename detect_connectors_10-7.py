# detect_offpage_connectors.py
# ─────────────────────────────────────────────────────────────────────────────
# Standalone script to detect off-page connectors (chevron/pentagon boundary
# arrows) in a P&ID drawing and draw green bounding boxes around them.
#
# Usage:
#   python detect_offpage_connectors.py --image path/to/pid.png
#   python detect_offpage_connectors.py --image path/to/pid.png --dpi 300
#   python detect_offpage_connectors.py --image path/to/pid.png --debug
#
# Output:
#   outputs/offpage_connectors_result.png  ← annotated image (green boxes)
#   outputs/offpage_connectors_result.json ← structured results

import os
import cv2
import json
import argparse
import numpy as np
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# Helper — pixel density fallback for direction
# ═══════════════════════════════════════════════════════════════════════════════

def _pixel_direction(closed_binary, cnt, bx, bw):
    """
    Fallback direction detection using ink pixel density.
    The blunt (base) end has MORE ink than the pointed tip.
    So: denser half = base → direction points AWAY from it.
    """
    mask = np.zeros_like(closed_binary)
    cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
    left_px  = int(np.sum(mask[:, bx           : bx + bw // 2] > 0))
    right_px = int(np.sum(mask[:, bx + bw // 2 : bx + bw     ] > 0))
    if left_px  > right_px * 1.15: return "RIGHT"
    if right_px > left_px  * 1.15: return "LEFT"
    return "UNKNOWN"


# ═══════════════════════════════════════════════════════════════════════════════
# Main detection function
# ═══════════════════════════════════════════════════════════════════════════════

def detect_offpage_connectors(
    image_path : str,
    output_dir : str  = "outputs",
    # ── Size filters (in pixels² — scale with DPI) ──────────────────────────
    min_area   : int   = 300,
    max_area   : int   = 120000,
    # ── Shape filters ────────────────────────────────────────────────────────
    aspect_min : float = 1.0,    # connector is wider than tall
    aspect_max : float = 8.0,
    # ── Convexity ────────────────────────────────────────────────────────────
    conv_min   : float = 0.60,   # relaxed for scanned/PDF drawings
    # ── Debug ────────────────────────────────────────────────────────────────
    debug      : bool  = False,
) -> dict:
    """
    Detect off-page connector chevrons in a P&ID image.

    Pipeline:
      1. Grayscale → Otsu binarize (ink = white)
      2. Morphological close (patches scan gaps)
      3. Find all external contours
      4. Filter by: area → aspect ratio → polygon vertices → convexity
      5. Determine direction from vertex geometry (fallback: pixel density)
      6. Draw green boxes + labels, save PNG + JSON

    Returns dict with keys: connectors, total, outgoing, incoming, unknown
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load image ────────────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis     = img.copy()
    H, W    = gray.shape

    # ── Stage 1: Binarize ────────────────────────────────────────────────────
    # THRESH_BINARY_INV  → ink becomes white (255), background black (0)
    # THRESH_OTSU        → auto-calculate best threshold for this image
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ── Stage 2: Morphological close ─────────────────────────────────────────
    # Bridges tiny gaps in connector outlines from scan noise
    k3     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3, iterations=2)

    if debug:
        debug_path = os.path.join(output_dir, "debug_binary.png")
        cv2.imwrite(debug_path, closed)
        print(f"  [debug] binary image saved → {debug_path}")

    # ── Stage 3: Find all contours ────────────────────────────────────────────
    # RETR_EXTERNAL  → only outermost shapes (ignores text/holes inside connectors)
    # CHAIN_APPROX_SIMPLE → stores only corner points (efficient for polygons)
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if debug:
        print(f"  [debug] total contours found: {len(contours)}")

    # ── Stage 4: Filters ──────────────────────────────────────────────────────
    # Multiple epsilon fractions for polygon approximation
    # Tries loosest→tightest until we get a 4-7 vertex shape (pentagon range)
    EPSILON_FRACS = [0.03, 0.05, 0.08, 0.12, 0.16]

    drop_area = drop_aspect = drop_poly = drop_convex = 0
    connectors = []
    conn_id    = 0

    for cnt in contours:

        # Filter 1 — Area
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            drop_area += 1
            continue

        # Filter 2 — Aspect ratio
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        aspect = bw / bh
        if not (aspect_min <= aspect <= aspect_max):
            drop_aspect += 1
            continue

        # Filter 3 — Polygon vertex count (multi-epsilon sweep)
        peri   = cv2.arcLength(cnt, True)
        approx = None
        for eps_frac in EPSILON_FRACS:
            candidate = cv2.approxPolyDP(cnt, eps_frac * peri, True)
            if 4 <= len(candidate) <= 7:
                approx = candidate
                break
        if approx is None:
            drop_poly += 1
            if debug:
                vcounts = [len(cv2.approxPolyDP(cnt, f*peri, True))
                           for f in EPSILON_FRACS]
                print(f"  [debug] poly-fail  area={int(area):6d} "
                      f"aspect={aspect:.2f} verts={vcounts}")
            continue

        # Filter 4 — Convexity (solidity check)
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1:
            continue
        conv_ratio = area / hull_area
        if conv_ratio < conv_min:
            drop_convex += 1
            if debug:
                print(f"  [debug] convex-fail area={int(area):6d} "
                      f"aspect={aspect:.2f} conv={conv_ratio:.2f}")
            continue

        # ── Stage 5: Direction ─────────────────────────────────────────────
        pts         = approx.reshape(-1, 2)
        leftmost_x  = pts[:, 0].min()
        rightmost_x = pts[:, 0].max()
        tol_px      = max(bw * 0.12, 8)

        n_near_left  = int(np.sum(pts[:, 0] <= leftmost_x  + tol_px))
        n_near_right = int(np.sum(pts[:, 0] >= rightmost_x - tol_px))

        if   n_near_left  == 1 and n_near_right >= 2:
            direction = "LEFT"    # tip at left → incoming from right sheet
        elif n_near_right == 1 and n_near_left  >= 2:
            direction = "RIGHT"   # tip at right → outgoing to right sheet
        else:
            direction = _pixel_direction(closed, cnt, bx, bw)

        tip_x = int(leftmost_x) if direction == "LEFT" else int(rightmost_x)

        entry = {
            "id"        : f"OPC{conn_id:04d}",
            "direction" : direction,
            "label"     : "-> OUT" if direction == "RIGHT"
                          else "<- IN" if direction == "LEFT" else "? UNK",
            "bbox"      : {"x": int(bx), "y": int(by),
                           "w": int(bw), "h": int(bh)},
            "center"    : {"x": int(bx + bw//2), "y": int(by + bh//2)},
            "area_px"   : int(area),
            "n_vertices": len(approx),
            "conv_ratio": round(conv_ratio, 3),
            "tip_x"     : tip_x,
        }
        connectors.append(entry)
        conn_id += 1

        # ── Stage 6: Draw on image ─────────────────────────────────────────
        # Match the green colour from your reference output image
        BOX_COLOR   = (0, 220, 60)    # bright green (BGR)
        LABEL_COLOR = (0, 220, 60)

        # Outer bounding box
        cv2.rectangle(vis,
                      (bx - 2, by - 2), (bx + bw + 2, by + bh + 2),
                      BOX_COLOR, 2)

        # Small inner box at the tip end
        tip_half = max(6, bh // 3)
        cv2.rectangle(vis,
                      (tip_x - tip_half, by),
                      (tip_x + tip_half, by + bh),
                      BOX_COLOR, 2)

        # Direction label above the box
        lbl    = entry["label"]
        font_s = max(0.30, min(bh / 50.0, 0.52))
        lbl_y  = max(by - 4, 14)
        # Thin black shadow for readability
        cv2.putText(vis, lbl, (bx + 1, lbl_y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_s, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis, lbl, (bx, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_s, LABEL_COLOR, 1, cv2.LINE_AA)

        # Trace the detected polygon outline
        cv2.polylines(vis, [approx], True, BOX_COLOR, 1)

    # ── Debug summary ─────────────────────────────────────────────────────────
    if debug:
        print(f"\n  [debug] Filter summary:")
        print(f"    Total contours   : {len(contours)}")
        print(f"    Dropped (area)   : {drop_area}")
        print(f"    Dropped (aspect) : {drop_aspect}")
        print(f"    Dropped (polygon): {drop_poly}")
        print(f"    Dropped (convex) : {drop_convex}")
        print(f"    PASSED           : {len(connectors)}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    vis_path  = os.path.join(output_dir, f"{base_name}_connectors.png")
    json_path = os.path.join(output_dir, f"{base_name}_connectors.json")

    cv2.imwrite(vis_path, vis)

    result = {
        "source_image": image_path,
        "timestamp"   : datetime.now().isoformat(),
        "image_size"  : {"width": int(W), "height": int(H)},
        "parameters"  : {
            "min_area"  : min_area,
            "max_area"  : max_area,
            "aspect_min": aspect_min,
            "aspect_max": aspect_max,
            "conv_min"  : conv_min,
        },
        "connectors"  : connectors,
        "total"       : len(connectors),
        "outgoing"    : sum(1 for c in connectors if c["direction"] == "RIGHT"),
        "incoming"    : sum(1 for c in connectors if c["direction"] == "LEFT"),
        "unknown"     : sum(1 for c in connectors if c["direction"] == "UNKNOWN"),
    }
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  ✓ Found {len(connectors)} off-page connectors")
    print(f"    Outgoing (->): {result['outgoing']}")
    print(f"    Incoming (<-): {result['incoming']}")
    print(f"    Unknown      : {result['unknown']}")
    print(f"  Image → {vis_path}")
    print(f"  JSON  → {json_path}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect off-page connector chevrons in a P&ID drawing"
    )
    parser.add_argument("--image",   required=True,
                        help="Path to the P&ID PNG image")
    parser.add_argument("--output",  default="outputs",
                        help="Output directory (default: outputs/)")
    parser.add_argument("--dpi",     type=int, default=150,
                        help="DPI of the image (default: 150). "
                             "Area thresholds scale automatically.")
    parser.add_argument("--debug",   action="store_true",
                        help="Print per-filter drop counts and save binary image")
    parser.add_argument("--min-area",   type=int,   default=None)
    parser.add_argument("--max-area",   type=int,   default=None)
    parser.add_argument("--aspect-min", type=float, default=1.0)
    parser.add_argument("--aspect-max", type=float, default=8.0)
    parser.add_argument("--conv-min",   type=float, default=0.60)
    args = parser.parse_args()

    # Scale area thresholds with DPI
    # Base values tuned at 150 DPI; area scales as (DPI/150)²
    scale = (args.dpi / 150) ** 2
    min_area = args.min_area if args.min_area is not None else int(300  * scale)
    max_area = args.max_area if args.max_area is not None else int(120000 * scale)

    print(f"\nOff-Page Connector Detection")
    print(f"  Image      : {args.image}")
    print(f"  DPI        : {args.dpi}")
    print(f"  Area range : {min_area} – {max_area} px²")
    print(f"  Aspect     : {args.aspect_min} – {args.aspect_max}")
    print(f"  Convexity  : ≥ {args.conv_min}")
    print()

    detect_offpage_connectors(
        image_path = args.image,
        output_dir = args.output,
        min_area   = min_area,
        max_area   = max_area,
        aspect_min = args.aspect_min,
        aspect_max = args.aspect_max,
        conv_min   = args.conv_min,
        debug      = args.debug,
    )