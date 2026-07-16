"""
Off-page connector detector for P&ID drawings (Aker Kvaerner / PPG style).

Target shape: a wide "flag" — a rectangle with one short end drawn as a
triangular point (5-8 vertex polygon after cleanup) — with a near-circular
index-number badge nested at the *non-pointed* end, and a dash-coded tag
(e.g. "07A-10051") printed inside the flag body.

Pipeline:
  1. Binarize + morphologically close to bridge anti-aliased line breaks.
  2. cv2.findContours with a 2-level hierarchy (RETR_CCOMP).
  3. Flag candidates = parent contours with a wide aspect ratio and an
     area inside an auto-calibrated size band for this sheet.
  4. For each flag, its index circle = the child contour with the
     highest circularity (4*pi*Area/Perimeter^2) above a threshold.
  5. Tip point = the flag-contour point farthest from the circle center
     (the circle always sits at the flat/back end, the point is the
     opposite extremity) -> gives a compass direction.
  6. OCR the circle crop for the index number and the remaining flag
     body (circle masked out) for the dash-coded tag, with a regex
     fallback/cleanup pass since P&ID fonts + JPEG artifacts confuse
     tesseract on isolated digits.

Usage:
    python detect_offpage_connectors.py input.png [--out out_dir] [--debug]
"""

import argparse
import json
import os
import re
from collections import Counter

import cv2
import numpy as np
import pytesseract

TAG_RE = re.compile(r"\d{1,3}[A-Z]?\s*-\s*\d{4,6}")
DIGITS_RE = re.compile(r"\d+")


def circularity(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return 0.0
    return float(4 * np.pi * area / (peri * peri))


def compass_direction(dx, dy):
    ang = np.degrees(np.arctan2(dy, dx))  # image coords: +y is down
    dirs = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
    idx = int(((ang + 22.5) % 360) // 45)
    return dirs[idx]


def binarize(gray):
    # Otsu as a baseline, but P&ID line art is thin/light on some scans,
    # so also try a fixed high threshold and keep whichever yields more ink
    # in a sane range (avoids picking up scanner noise as "ink").
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, fixed = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    frac_otsu = cv2.countNonZero(otsu) / otsu.size
    frac_fixed = cv2.countNonZero(fixed) / fixed.size
    # ink should be a small minority of the page (text/lines on white)
    return otsu if 0.005 < frac_otsu < 0.35 else fixed


def median_char_height(gray):
    """OCR-based scale calibration: robust to sheet resolution, independent
    of any single symbol's size. Downscales for speed since only the
    typical text height is needed, not exact word boxes."""
    h_img, w_img = gray.shape
    scale = min(1.0, 2000 / w_img)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    data = pytesseract.image_to_data(small, output_type=pytesseract.Output.DICT)
    heights = [h for h, txt in zip(data["height"], data["text"]) if txt.strip() and 4 < h < 60]
    if not heights:
        return max(10, h_img / 220)  # fallback heuristic
    return float(np.median(heights)) / scale


def find_circle_badges(gray, diam_lo, diam_hi):
    """Stage 1: locate near-circular, near-square badges of the expected
    size anywhere on the sheet. Cheap, and avoids the global contour-merge
    problems that whole-flag detection runs into when a badge happens to
    touch an unrelated dashed process line."""
    th = binarize(gray)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    badges = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (diam_lo <= w <= diam_hi and diam_lo <= h <= diam_hi):
            continue
        sq = min(w, h) / max(w, h)
        if sq < 0.6:
            continue
        if circularity(c) < 0.55:
            continue
        badges.append((x, y, w, h))

    # de-duplicate near-identical boxes (nested contours of the same badge)
    deduped = []
    for b in sorted(badges, key=lambda b: -(b[2] * b[3])):
        bx, by, bw, bh = b
        bc = (bx + bw / 2, by + bh / 2)
        if any(np.hypot(bc[0] - (dx + dw / 2), bc[1] - (dy + dh / 2)) < max(bw, dw) * 0.5
               for dx, dy, dw, dh in deduped):
            continue
        deduped.append(b)
    return deduped


def find_flag_around_badge(gray, badge_bbox, search_mult=16.0):
    """Stage 2: re-threshold a small ROI around a candidate badge and look
    for the wide flag polygon it sits on -- either to its left or right,
    since a connector can point either direction."""
    bx, by, bw, bh = badge_bbox
    cx, cy = bx + bw / 2, by + bh / 2
    win_w, win_h = bw * search_mult, bh * 1.6
    x0, y0 = int(max(0, cx - win_w / 2)), int(max(0, cy - win_h / 2))
    x1, y1 = int(min(gray.shape[1], cx + win_w / 2)), int(min(gray.shape[0], cy + win_h / 2))
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    th = binarize(roi)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    badge_x0, badge_y0 = bx - x0, by - y0
    badge_x1, badge_y1 = badge_x0 + bw, badge_y0 + bh
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < (bw * bh) * 1.5:  # must be noticeably bigger than the badge alone
            continue
        x, y, w, h = cv2.boundingRect(c)
        long_side, short_side = max(w, h), max(min(w, h), 1)
        if long_side / short_side < 2.2:
            continue

        # the flag must sit in the same row as the badge (strong vertical
        # overlap) and be horizontally adjacent to or overlapping it --
        # NOT necessarily contain the badge's center point, since the
        # badge and the flag body are frequently separate touching
        # contours rather than one merged shape
        vert_overlap = max(0, min(y + h, badge_y1) - max(y, badge_y0))
        if vert_overlap < 0.5 * min(h, bh):
            continue
        horiz_gap = max(0, max(x, badge_x0) - min(x + w, badge_x1))
        if horiz_gap > bw * 0.6:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True).reshape(-1, 2)
        if not (4 <= len(approx) <= 10):
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_area if hull_area else 0
        if solidity < 0.55:
            continue
        if best is None or area < best[0]:  # smallest qualifying = tightest fit
            best = (area, c, approx, (x, y, w, h))

    if best is None:
        return None
    _, c, approx, (x, y, w, h) = best
    # translate contour + bbox back to full-image coordinates
    c_full = c + np.array([x0, y0])
    approx_full = approx + np.array([x0, y0])
    return {
        "contour": c_full,
        "approx": approx_full,
        "bbox": (x + x0, y + y0, w, h),
        "circle_bbox": badge_bbox,
    }


def find_flags(gray, diam_lo, diam_hi):
    badges = find_circle_badges(gray, diam_lo, diam_hi)
    flags = []
    for b in badges:
        flag = find_flag_around_badge(gray, b)
        if flag is None:  # retry wider/taller in case the tight window clipped the flag
            flag = find_flag_around_badge(gray, b, search_mult=24.0)
        if flag is not None:
            flags.append(flag)
    return flags


def tip_and_direction(flag):
    x, y, w, h = flag["bbox"]
    cx, cy, cw, ch = flag["circle_bbox"]
    circle_center = np.array([cx + cw / 2, cy + ch / 2])
    box_center = np.array([x + w / 2, y + h / 2])

    pts = flag["contour"].reshape(-1, 2).astype(float)
    dists = np.linalg.norm(pts - circle_center, axis=1)
    tip = pts[np.argmax(dists)]

    dx, dy = tip - box_center
    direction = compass_direction(dx, dy)
    return tuple(tip.astype(int)), direction


def ocr_tag(gray, flag):
    x, y, w, h = flag["bbox"]
    cx, cy, cw, ch = flag["circle_bbox"]
    pad = 4

    # text lives strictly between the circle badge and the start of the
    # pointed tip (whichever side that tip is on) -- excluding the tip
    # avoids its converging line art being misread as stray digits, and
    # excluding the circle avoids the index number leaking into the tag
    circle_on_left = (cx - x) < w / 2
    if circle_on_left:
        text_x0 = max(0, (cx - x) + cw + pad)
        text_x1 = int(w * 0.92)
    else:
        text_x0 = int(w * 0.08)
        text_x1 = min(w, (cx - x) - pad)
    if text_x1 <= text_x0:
        text_x0, text_x1 = 0, w
    crop = gray[max(0, y - pad): y + h + pad, x + text_x0: x + text_x1].copy()
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, crop_th = cv2.threshold(crop, 180, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(
        crop_th, config="--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
    )
    text = text.strip().replace(" ", "")
    m = TAG_RE.search(text)
    return m.group(0).replace(" ", "") if m else (text or None)


def ocr_index_number(gray, flag):
    cx, cy, cw, ch = flag["circle_bbox"]
    # crop to the circle's inscribed interior (avoid the ring stroke and
    # any adjacent box edge -- both confuse tesseract's segmentation badly
    # on a glyph this small)
    mx, my = int(cw * 0.16), int(ch * 0.16)
    crop = gray[cy + my: cy + ch - my, cx + mx: cx + cw - mx]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    _, crop_th = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # small OCR ensemble: circled digits are adversarial for tesseract, and
    # different page-segmentation modes fail on different examples, so vote
    candidates = []
    for psm in (6, 7, 8, 13):
        t = pytesseract.image_to_string(
            crop_th, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789"
        ).strip()
        digits = DIGITS_RE.search(t)
        if digits:
            candidates.append(digits.group(0))
    if not candidates:
        return None
    counts = Counter(candidates)
    best = max(candidates, key=lambda d: (len(d), counts[d]))
    return int(best)


def detect(image_path, debug_dir=None):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # auto-calibrate the expected index-badge diameter off the sheet's own
    # OCR text height (resolution-independent; badges run ~2.5-3.5x the
    # median character height on every sheet we've seen)
    char_h = median_char_height(gray)
    diam_lo, diam_hi = char_h * 2.0, char_h * 4.5

    flags = find_flags(gray, diam_lo=diam_lo, diam_hi=diam_hi)

    results = []
    for k, flag in enumerate(flags):
        tip, direction = tip_and_direction(flag)
        tag = ocr_tag(gray, flag)
        idx_num = ocr_index_number(gray, flag)
        x, y, w, h = flag["bbox"]
        results.append(
            {
                "id": k,
                "bbox": [int(x), int(y), int(w), int(h)],
                "circle_bbox": [int(v) for v in flag["circle_bbox"]],
                "tip_point": [int(tip[0]), int(tip[1])],
                "tip_direction": direction,
                "index_number": idx_num,
                "tag_code": tag,
                "needs_review": tag is None or idx_num is None,
            }
        )

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        vis = img.copy()
        for r in results:
            x, y, w, h = r["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 3)
            cx, cy, cw, ch = r["circle_bbox"]
            cv2.rectangle(vis, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 2)
            tx, ty = r["tip_point"]
            cv2.circle(vis, (tx, ty), 8, (0, 0, 255), -1)
            label = f'{r["index_number"]}: {r["tag_code"]} ({r["tip_direction"]})'
            cv2.putText(vis, label, (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        out_img = os.path.join(debug_dir, "annotated_" + os.path.basename(image_path))
        cv2.imwrite(out_img, vis)
        with open(os.path.join(debug_dir, "detections.json"), "w") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="15-7_output", help="output directory for annotated image + JSON ")
    args = ap.parse_args()
    res = detect(args.image, debug_dir=args.out)
    print(json.dumps(res, indent=2))
    print(f"\n{len(res)} connector(s) found. Annotated image + JSON in {args.out}/")