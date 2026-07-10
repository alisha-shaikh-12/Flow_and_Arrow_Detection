"""
Off-page connector detector for P&ID drawings.

Pipeline:
 1. Binarize the drawing and find candidate contours shaped like the
    wide, low, pentagon/chevron "off-page connector" symbol.
 2. Merge contours that are horizontally adjacent (handles connectors whose
    outline + reference-number tag get split into two contours by the
    binarizer).
 3. For each candidate, expand the box upward to capture the descriptive
    text that normally sits above the symbol, then OCR that region.
 4. Keep only candidates whose OCR text contains a reference-drawing tag
    (e.g. "G-30564") or an explicit FROM/TO flow keyword - this rejects
    false positives such as instrument bubbles (PI/TI/LT circles) that
    happen to share a similar bounding-box footprint.
 5. Classify each kept connector as INFLOW or OUTFLOW from its text, and
    record which way the chevron point is oriented as a secondary signal.
 6. Draw color-coded bounding boxes + labels on an output PNG and write a
    JSON file with one record per connector.
"""

import cv2
import numpy as np
import pytesseract
import re
import json
import argparse
import os

# ---- tunable geometry parameters (calibrated against the sample P&ID) ----
MIN_AREA = 800
MAX_AREA = 20000
MIN_W = 55
MIN_H, MAX_H = 15, 70
MIN_AR, MAX_AR = 1.8, 9.0
MERGE_GAP_PX = 25          # max horizontal gap to merge two adjacent contours
TEXT_MARGIN_ABOVE = 130    # px to look above the shape for description text
TEXT_MARGIN_SIDE = 40

TAG_RE = re.compile(r'[GT][K]?[\s\-–—]{0,2}\d{4,6}')
# Stricter pattern used against the dedicated tag-only OCR pass: this sheet's
# reference tags are 5 digits, and trailing extra digits are common OCR noise
# from the chevron's border line, so we only keep the first 5 after the dash.
TAG_STRICT_RE = re.compile(r'([GT][K]?)?\s*[\-–—]\s*(\d{5})')
EQUIP_RE = re.compile(r'\b(PU|TK|HE)[\s\-]?\d{2,4}[A-Z]?(?:/[A-Z0-9]+)*\b', re.IGNORECASE)
FROM_RE = re.compile(r'\bFROM\b', re.IGNORECASE)
TO_RE = re.compile(r'\bTO\b', re.IGNORECASE)
SUPPLY_RE = re.compile(r'\bSUPPLY\b', re.IGNORECASE)
RETURN_RE = re.compile(r'\bRETURN\b', re.IGNORECASE)

# Below this y-coordinate the sheet is title block / revision table, not
# drawing content -- any candidate shape found there is a false positive.
TITLE_BLOCK_Y = 1950


def find_candidate_boxes(gray):
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h) if h else 0
        if w < MIN_W or not (MIN_H < h < MAX_H) or not (MIN_AR < ar < MAX_AR):
            continue
        boxes.append([x, y, w, h])
    return boxes


def merge_adjacent(boxes):
    """
    Merge horizontally-adjacent boxes on (roughly) the same text line.
    Returns a list of (merged_box, shape_box) pairs, where shape_box is the
    single raw sub-box that sits lowest (closest to the pipe/text below it)
    -- that's the actual chevron/arrow containing the reference tag, as
    opposed to a plain text fragment that got merged alongside it.
    """
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged = []
    used = [False] * len(boxes)

    for i, b in enumerate(boxes):
        if used[i]:
            continue
        x, y, w, h = b
        cur = [x, y, x + w, y + h]
        group = [b]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, b2 in enumerate(boxes):
                if used[j]:
                    continue
                x2, y2, w2, h2 = b2
                bx0, by0, bx1, by1 = x2, y2, x2 + w2, y2 + h2
                y_overlap = min(cur[3], by1) - max(cur[1], by0)
                min_h = min(cur[3] - cur[1], by1 - by0)
                same_row = min_h > 0 and y_overlap / min_h > 0.4
                gap = max(cur[0] - bx1, bx0 - cur[2])
                if same_row and gap < MERGE_GAP_PX:
                    cur = [min(cur[0], bx0), min(cur[1], by0),
                           max(cur[2], bx1), max(cur[3], by1)]
                    used[j] = True
                    changed = True
                    group.append(b2)
        shape_box = max(group, key=lambda gb: gb[1])  # lowest (largest y)
        merged.append(([cur[0], cur[1], cur[2] - cur[0], cur[3] - cur[1]], shape_box))
    return merged


def find_text_block_above(bin_img, x, y, w, h, max_up=180, side_pad=45,
                           blank_gap=10, ink_thresh=8):
    """
    Scan upward from the shape's top edge to find the contiguous block of
    descriptive text sitting above it, stopping at the first real blank
    gap (a row-run with near-zero ink) so we don't bleed into an unrelated
    label further up the page. `ink_thresh` filters out the constant few
    pixels contributed by a vertical pipe/border line passing through the
    strip.
    """
    img_h, img_w = bin_img.shape
    x0 = max(0, x - side_pad)
    x1 = min(img_w, x + w + side_pad)
    top = max(0, y - max_up)
    strip = bin_img[top:y, x0:x1]
    if strip.shape[0] == 0:
        return x0, y, x1, y
    row_ink = (strip > 0).sum(axis=1)
    rows = row_ink[::-1]  # bottom (closest to shape) -> top

    started = False
    blank_run = 0
    cutoff = len(rows)
    for i, v in enumerate(rows):
        if v > ink_thresh:
            started = True
            blank_run = 0
        elif started:
            blank_run += 1
            if blank_run >= blank_gap:
                cutoff = i - blank_run + 1
                break
    y_start = y - max(cutoff, 0)
    return x0, max(top, y_start), x1, y


def ocr_region(gray, bin_img, x, y, w, h, img_w, img_h):
    x0, y0, x1, y1 = find_text_block_above(bin_img, x, y, w, h)
    y1 = min(img_h, y + h)  # include the shape itself too
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return ""
    roi = cv2.resize(roi, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(roi, config='--psm 6')
    return text.strip()


def ocr_tag(gray, x, y, w, h):
    """
    Dedicated OCR pass for the reference-drawing tag printed inside the
    chevron itself. The tag text sits just above the chevron's bottom
    border line, which otherwise confuses OCR when included, so several
    bottom-trim ratios are tried and the first one that yields a clean
    "<letter>-<5 digits>" match is used.
    """
    for trim_frac in (0.30, 0.35, 0.39, 0.25, 0.20):
        trim_bot = max(1, int(h * trim_frac))
        roi = gray[y:y + h - trim_bot, x:x + w]
        if roi.size == 0:
            continue
        roi = cv2.resize(roi, None, fx=6.0, fy=6.0, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(
            roi, config='--psm 7 -c tessedit_char_whitelist=GTK0123456789-'
        ).strip()
        m = TAG_STRICT_RE.search(text)
        if m:
            prefix = (m.group(1) or "G").upper()
            digits = m.group(2)
            return f"{prefix}-{digits}"
    return None


def classify_direction(text):
    has_from = bool(FROM_RE.search(text))
    has_to = bool(TO_RE.search(text))
    if has_from and not has_to:
        return "inflow"
    if has_to and not has_from:
        return "outflow"
    if has_from and has_to:
        # e.g. stray OCR noise; prefer whichever appears first in the text
        return "inflow" if FROM_RE.search(text).start() < TO_RE.search(text).start() else "outflow"
    # No explicit FROM/TO -- fall back to SUPPLY/RETURN convention:
    # a "supply" line sends fluid out to the off-page destination,
    # a "return" line brings fluid back in from it.
    if SUPPLY_RE.search(text):
        return "outflow"
    if RETURN_RE.search(text):
        return "inflow"
    return "unknown"


def point_orientation(gray, x, y, w, h):
    """Rough left/right chevron-point check via row-wise ink profile."""
    roi = gray[y:y + h, x:x + w]
    _, th = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)
    col_ink = th.sum(axis=0)
    if col_ink.sum() == 0:
        return "unknown"
    left_ink = col_ink[: w // 4].mean()
    right_ink = col_ink[-w // 4:].mean()
    # the pointed end tends to have less ink coverage than the flat end
    return "points_left" if left_ink < right_ink else "points_right"


def process(image_path, out_png, out_json):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    raw_boxes = find_candidate_boxes(gray)
    merged_pairs = merge_adjacent(raw_boxes)

    results = []
    annotated = img.copy()

    conn_id = 1
    for (x, y, w, h), (sx, sy, sw, sh) in merged_pairs:
        if y >= TITLE_BLOCK_Y:
            continue  # title block / revision table, not drawing content

        text = ocr_region(gray, bin_img, x, y, w, h, img_w, img_h)
        tag_match = TAG_RE.search(text)
        has_keyword = bool(
            FROM_RE.search(text) or TO_RE.search(text)
            or SUPPLY_RE.search(text) or RETURN_RE.search(text)
        )

        if not tag_match and not has_keyword:
            continue  # reject false positives (instrument bubbles, stray boxes, etc.)

        direction = classify_direction(text)
        orientation = point_orientation(gray, sx, sy, sw, sh)

        # prefer the dedicated tag-only OCR pass (much more accurate for the
        # text sitting inside the chevron); fall back to the looser match
        # found in the description-block OCR text.
        tag = ocr_tag(gray, sx, sy, sw, sh)
        if not tag and tag_match:
            tag = tag_match.group(0).replace(' ', '').upper()

        # first non-empty line(s) as the human-readable description
        desc_lines = [l.strip() for l in text.splitlines() if l.strip()]
        description = " ".join(desc_lines[:-1]) if len(desc_lines) > 1 else (desc_lines[0] if desc_lines else "")
        equip_match = EQUIP_RE.search(text)
        equipment_tag = equip_match.group(0).upper() if equip_match else None

        color = (0, 165, 0) if direction == "inflow" else (0, 0, 220) if direction == "outflow" else (150, 150, 150)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        label = f"#{conn_id} {direction.upper()}"
        (tw, th_) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        ly = max(0, y - 8)
        cv2.rectangle(annotated, (x, ly - th_ - 6), (x + tw + 6, ly + 2), color, -1)
        cv2.putText(annotated, label, (x + 3, ly - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        results.append({
            "id": conn_id,
            "bbox_xywh": [int(x), int(y), int(w), int(h)],
            "reference_tag": tag,
            "description": description,
            "equipment_tag": equipment_tag,
            "raw_ocr_text": text,
            "direction": direction,
            "chevron_point_orientation": orientation,
        })
        conn_id += 1

    cv2.imwrite(out_png, annotated)
    with open(out_json, 'w') as f:
        json.dump({
            "source_image": os.path.basename(image_path),
            "image_size": {"width": img_w, "height": img_h},
            "connector_count": len(results),
            "connectors": results,
        }, f, indent=2)

    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    args = ap.parse_args()

    # -------------------------------
    # Create output folder
    # -------------------------------
    output_dir = "8_07"
    os.makedirs(output_dir, exist_ok=True)

    # Base filename of input image
    image_name = os.path.splitext(os.path.basename(args.image))[0]

    # Output filenames
    out_png = os.path.join(output_dir, f"{image_name}_annotated.jpg")
    out_json = os.path.join(output_dir, f"{image_name}.json")

    # Run detection
    res = process(args.image, out_png, out_json)

    print(f"\nFound {len(res)} connectors")
    print(f"Annotated image : {out_png}")
    print(f"JSON output     : {out_json}")

    for r in res:
        print(r["id"], r["reference_tag"], r["direction"], r["description"][:60])