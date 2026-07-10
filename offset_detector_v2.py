"""
Off-page connector detector for P&ID drawings - v4.

*** Why v3 failed on real mixed-vendor sheets ***
Tested against two more sheets (3024px-wide Technip/Westlake sheet, and an
8000px-wide PPG/Aker Kvaerner sheet) and v3 broke in three specific ways:

1. All the size/shape filters were hardcoded as fractions of image
   dimensions, tuned against ONE sheet. A 3024px sheet and an 8000px sheet
   don't share a "0.008 of image height" tag size - that number only meant
   something on the original sheet it was measured from.
2. Opening up top/bottom margins (v3's new default) started catching
   title-block cells, duty-spec boxes, and revision-table rows - none of
   which are off-page connectors - because those live in the same margin
   band and nothing distinguished them geometrically.
3. PPG/Aker Kvaerner draws off-page connectors as an oval/circle + small
   arrow, not the chevron pentagon Technip uses. The chevron-tuned contour
   filter doesn't fire on it at all, or fires on the wrong nearby shape.

*** What's different in v4 ***

1. AUTO-CALIBRATED SIZE THRESHOLDS. Before looking for candidate shapes,
   v4 runs one whole-page OCR pass and takes the median word height as a
   "reference character height" for THIS sheet. All box-size filters are
   then expressed as multiples of that (e.g. "candidate height is 2-6x a
   character tall"), so the same config numbers hold whether the sheet is
   3000px or 8000px wide. Verified on both test sheets: median char height
   came out ~8px and ~23px respectively - very different absolute numbers,
   almost the same fraction of page width - which is exactly the kind of
   thing that should be measured per-sheet, not hardcoded.

2. WORD-ANCHORED LABEL SEARCH, not band-cropping. v3 OCR'd a big rectangular
   band above/below each candidate box, which works fine when a tag sits in
   open space but mashes multiple table cells together into garbage text
   when a false-positive candidate sits inside a dense title block or
   spec table. v4 instead runs ONE image_to_data() pass over the whole page
   up front, giving every word its own bounding box, then for each
   candidate box just gathers the individual OCR words whose boxes are
   actually near it. No band, no bleed.

3. A REQUIRED "this looks like a real tag" text check. A genuine off-page
   connector always has a structured drawing/line reference near it
   (e.g. "G-30277A", "07A-10235", "LIC-30370"). Table cells, dates, and
   revision numbers don't match that pattern. Candidates with no such
   token nearby are dropped before they ever reach the direction logic -
   this is what kills most of the title-block/spec-box false positives
   v3 was producing, without needing a hand-maintained exclude list.

4. AUTO-DETECTED "DENSE TEXT" EXCLUSION ZONES. Title blocks and revision
   tables are visually distinctive even without knowing their exact
   layout: they're regions where OCR words are packed far more densely
   than anywhere else on the sheet. v4 grids the whole-page word list,
   flags cells above a density threshold, and merges adjacent flagged
   cells into exclusion rectangles - no per-template hardcoding needed.
   You can also pass explicit --exclude-rect values if you already know a
   template's title-block position and want to guarantee it's skipped.

5. TWO SHAPE FAMILIES, not one. Alongside the v3 chevron/tip detector
   (still used for Technip-style tags), v4 adds a circularity-based
   detector for the oval/circle-plus-arrow style. Both shape families feed
   the same FROM/TO text logic, since that part of the convention (the
   words "FROM"/"TO" near the tag) held up across both vendors in testing.
   Geometric tip-direction (the "point = flow direction" rule from v3)
   is ONLY applied to chevron-shaped tags - it hasn't been verified for the
   oval style, so oval tags rely on text alone for now.

*** Still-open limitations, be aware of these ***
- Oval/circle tags get NO geometric direction fallback yet - if their
  FROM/TO text is unreadable, they'll come back "unknown" rather than a
  guess. Extending geometry to that shape family (e.g. detecting which
  side the small triangular arrow sits on) is a reasonable next step but
  needs its own sample-driven validation the way the chevron rule got.
- The dense-region exclusion is a heuristic, not a guarantee. On a sheet
  with an unusually text-dense process area (e.g. a legend/notes block
  outside the title block), it can over-exclude. Check the
  "excluded_regions" field in the JSON output and use --exclude-rect / a
  lower/higher --density-thresh if it's cutting out real connectors or
  letting a table through.
- Still only validated end-to-end on two sheets. Treat the accuracy
  numbers from evaluate_accuracy.py, not from reading the annotated image,
  as the basis for any finalize/don't-finalize decision.

Usage:
    python offset_detector_v4.py input.png --out-dir results/
    python offset_detector_v4.py input.png --exclude-rect 0.75,0.85,1.0,1.0
"""

import argparse
import json
import os
import re

import cv2
import numpy as np
import pytesseract


class Config:
    # ---- where to look ----
    edge_margin_frac = 0.16
    search_sides = ("left", "right", "top", "bottom")

    # ---- binarization: Otsu instead of a fixed threshold, since scan
    # brightness/contrast varies a lot across vendors/scanners ----
    close_kernel = 3
    close_iters = 1

    # ---- auto-calibrated candidate size filter (multiples of the sheet's
    # own median OCR word height, NOT fractions of image size - see docstring) ----
    min_height_mult = 1.6
    max_height_mult = 7.0
    min_width_mult = 3.0
    max_width_mult = 35.0
    min_aspect = 1.15
    max_aspect = 12.0

    # ---- text-based "is this a real tag" gate ----
    # matches things like G-30277A, 07A-10235, LIC-30370, TK-178A
    tag_pattern = re.compile(r"\b[0-9]{0,3}[A-Z]{1,4}-\d{2,6}[A-Z]{0,2}\b")
    require_tag_pattern_nearby = True
    tag_pattern_search_mult = 4.0   # search window, in units of box height

    # ---- FROM/TO detection over nearby OCR words ----
    label_search_mult_vertical = 3.0    # how far above/below to gather words, in box-heights
    label_search_mult_horizontal = 4.0  # how far left/right, in box-heights
    fuzzy_max_distance = 1
    from_keywords = ("FROM",)
    to_keywords = ("TO", "T0")

    # ---- exclude_keywords still checked as a belt-and-suspenders backstop ----
    exclude_keywords = ("ISSUED", "BULLETIN", "REVISION", "RECORD", "CONSTRUCTION",
                         "CHK", "APPD", "CHG", "REV.", "DESCRIPTION", "DRAWN",
                         "CHECKED", "APPROVED", "PROPERTY OF", "SCALE")

    # ---- auto-detected dense-text exclusion zones (title blocks, tables) ----
    auto_exclude_dense_regions = True
    dense_grid_frac = 0.04          # grid cell size as a fraction of image width
    dense_word_count_thresh = 14    # words in a cell -> flag as "table-like"
    user_exclude_rects = []         # list of (x0,y0,x1,y1) in 0-1 fractions, CLI-settable

    # ---- shape family 1: chevron / pentagon tip (same logic as v3) ----
    use_chevron_geometry = True
    tip_edge_frac = 0.06
    tip_ratio_max = 0.5
    blunt_ratio_min = 0.82
    min_shape_len_px = 20

    # ---- shape family 2: oval / circle + arrow (new in v4) ----
    circularity_min = 0.55   # 4*pi*area/perimeter^2 ; 1.0 = perfect circle

    # ---- line-attachment probe (soft signal only in v4, not a hard gate -
    # different shape families attach lines differently) ----
    probe_len_mult = 2.5     # in units of box height
    probe_band_px_mult = 0.5
    min_line_run_frac = 0.6

    # ---- deduplication ----
    dedup_iou_thresh = 0.4

    # ---- combination / confidence ----
    conflict_confidence = 0.45
    text_only_confidence = 0.85
    geometry_only_base_confidence = 0.55
    agree_confidence = 0.97


# --------------------------------------------------------------------------
# whole-page OCR, run once
# --------------------------------------------------------------------------
def ocr_words(gray, min_conf=25):
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config="--psm 11")
    words = []
    n = len(data["text"])
    for i in range(n):
        txt = data["text"][i].strip()
        conf = int(float(data["conf"][i])) if data["conf"][i] not in ("-1", "") else -1
        if not txt or conf < min_conf:
            continue
        words.append({
            "text": txt,
            "x": data["left"][i], "y": data["top"][i],
            "w": data["width"][i], "h": data["height"][i],
            "conf": conf,
        })
    return words


def estimate_char_height(words):
    heights = np.array([w["h"] for w in words if w["h"] > 0])
    if heights.size == 0:
        return 12.0  # fallback guess
    return float(np.median(heights))


# --------------------------------------------------------------------------
# binarization / margin mask
# --------------------------------------------------------------------------
def load_binary(gray, cfg: Config):
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cfg.close_iters > 0:
        k = np.ones((cfg.close_kernel, cfg.close_kernel), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=cfg.close_iters)
    return th


def margin_mask(shape, cfg: Config):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mw = int(w * cfg.edge_margin_frac)
    mh = int(h * cfg.edge_margin_frac)
    if "left" in cfg.search_sides:
        mask[:, :mw] = True
    if "right" in cfg.search_sides:
        mask[:, w - mw:] = True
    if "top" in cfg.search_sides:
        mask[:mh, :] = True
    if "bottom" in cfg.search_sides:
        mask[h - mh:, :] = True
    return mask


# --------------------------------------------------------------------------
# NEW in v4: auto-detected dense-text exclusion zones (title blocks/tables)
# --------------------------------------------------------------------------
def detect_dense_regions(words, img_w, img_h, cfg: Config):
    """Grid the page by word centers; cells with unusually many words get
    flagged and merged into exclusion rectangles. Title blocks and revision
    tables are dense grids of short text and stand out from normal drawing
    content this way, regardless of where the template puts them."""
    cell = max(1, int(img_w * cfg.dense_grid_frac))
    nx, ny = img_w // cell + 1, img_h // cell + 1
    grid = np.zeros((ny, nx), dtype=int)
    for w in words:
        cx, cy = w["x"] + w["w"] // 2, w["y"] + w["h"] // 2
        gx, gy = min(nx - 1, cx // cell), min(ny - 1, cy // cell)
        grid[gy, gx] += 1

    flagged = grid >= cfg.dense_word_count_thresh
    if not flagged.any():
        return []

    # merge adjacent flagged cells into connected blobs -> bounding rects
    flagged_u8 = flagged.astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(flagged_u8, connectivity=8)
    rects = []
    for label_id in range(1, n_labels):
        ys, xs = np.where(labels == label_id)
        x0, x1 = int(xs.min() * cell), int((xs.max() + 1) * cell)
        y0, y1 = int(ys.min() * cell), int((ys.max() + 1) * cell)
        rects.append((x0, y0, min(x1, img_w), min(y1, img_h)))
    return rects


def build_exclusion_mask(shape, words, cfg: Config, img_w, img_h):
    mask = np.zeros(shape[:2], dtype=bool)
    rects = []
    if cfg.auto_exclude_dense_regions:
        rects.extend(detect_dense_regions(words, img_w, img_h, cfg))
    for (fx0, fy0, fx1, fy1) in cfg.user_exclude_rects:
        rects.append((int(fx0 * img_w), int(fy0 * img_h), int(fx1 * img_w), int(fy1 * img_h)))
    for (x0, y0, x1, y1) in rects:
        mask[y0:y1, x0:x1] = True
    return mask, rects


# --------------------------------------------------------------------------
# candidate shape finder (auto-calibrated by char height, not raw image fractions)
# --------------------------------------------------------------------------
def find_candidate_shapes(binary_img, cfg: Config, valid_mask, char_h):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_h, max_h = cfg.min_height_mult * char_h, cfg.max_height_mult * char_h
    min_w, max_w = cfg.min_width_mult * char_h, cfg.max_width_mult * char_h

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0 or w == 0:
            continue
        cx, cy = x + w // 2, y + h // 2
        if not valid_mask[min(cy, valid_mask.shape[0] - 1), min(cx, valid_mask.shape[1] - 1)]:
            continue
        if not (min_h <= h <= max_h):
            continue
        if not (min_w <= w <= max_w):
            continue
        aspect = w / h
        if not (cfg.min_aspect <= aspect <= cfg.max_aspect):
            continue
        candidates.append({"contour": c, "bbox": (x, y, w, h)})
    return candidates


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def dedup_candidates(candidates, thresh):
    candidates = sorted(candidates, key=lambda c: c["bbox"][2] * c["bbox"][3], reverse=True)
    kept = []
    for cand in candidates:
        if all(iou(cand["bbox"], k["bbox"]) < thresh for k in kept):
            kept.append(cand)
    return kept


# --------------------------------------------------------------------------
# shape classification: chevron (tapered) vs oval/circle vs unclassified
# --------------------------------------------------------------------------
def _column_extents(mask_2d):
    w = mask_2d.shape[1]
    extents = np.zeros(w)
    for c in range(w):
        idx = np.where(mask_2d[:, c] > 0)[0]
        if idx.size:
            extents[c] = idx.max() - idx.min() + 1
    return extents


def geometric_tip_side(binary_img, bbox, cfg: Config):
    x, y, w, h = bbox
    crop = binary_img[y:y + h, x:x + w]
    if crop.size == 0:
        return "unknown", 0.0
    horizontal = w >= h
    axis_len = w if horizontal else h
    if axis_len < cfg.min_shape_len_px:
        return "unknown", 0.0

    work = crop if horizontal else crop.T
    extents = _column_extents(work)
    mid = extents[len(extents) // 3: 2 * len(extents) // 3]
    mid_h = np.median(mid) if mid.size else 0
    if mid_h <= 0:
        return "unknown", 0.0

    edge_n = max(2, int(axis_len * cfg.tip_edge_frac))
    start_ratio = float(np.mean(extents[:edge_n])) / mid_h
    end_ratio = float(np.mean(extents[-edge_n:])) / mid_h

    start_is_tip = start_ratio <= cfg.tip_ratio_max and end_ratio >= cfg.blunt_ratio_min
    end_is_tip = end_ratio <= cfg.tip_ratio_max and start_ratio >= cfg.blunt_ratio_min

    if start_is_tip:
        side = "left" if horizontal else "top"
        sharpness = min(1.0, (cfg.blunt_ratio_min - start_ratio) / cfg.blunt_ratio_min)
        return side, sharpness
    if end_is_tip:
        side = "right" if horizontal else "bottom"
        sharpness = min(1.0, (cfg.blunt_ratio_min - end_ratio) / cfg.blunt_ratio_min)
        return side, sharpness
    return "unknown", 0.0


def circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return float(4 * np.pi * area / (perimeter * perimeter))


def classify_shape(binary_img, contour, bbox, cfg: Config):
    """Returns (shape_type, tip_side, tip_sharpness). shape_type is one of
    'chevron', 'oval', 'unclassified'."""
    circ = circularity(contour)
    x, y, w, h = bbox
    aspect = w / h
    if circ >= cfg.circularity_min and aspect <= 1.6:
        return "oval", "unknown", 0.0

    if cfg.use_chevron_geometry:
        tip_side, sharp = geometric_tip_side(binary_img, bbox, cfg)
        if tip_side != "unknown":
            return "chevron", tip_side, sharp

    return "unclassified", "unknown", 0.0


def geo_flow_from_tip(tip_side, edge, into_page, out_of_page):
    if tip_side == "unknown" or edge == "unknown":
        return "unknown"
    if tip_side == out_of_page:
        return "outflow"
    if tip_side == into_page:
        return "inflow"
    return "unknown"


def page_edge(bbox, img_w, img_h, cfg: Config):
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    dists = {}
    if "left" in cfg.search_sides:
        dists["left"] = cx
    if "right" in cfg.search_sides:
        dists["right"] = img_w - cx
    if "top" in cfg.search_sides:
        dists["top"] = cy
    if "bottom" in cfg.search_sides:
        dists["bottom"] = img_h - cy
    return min(dists, key=dists.get) if dists else "unknown"


# --------------------------------------------------------------------------
# NEW in v4: word-anchored text gathering (replaces v3's band-crop OCR)
# --------------------------------------------------------------------------
def words_near(words, bbox, v_mult, h_mult):
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    win_h, win_w = v_mult * h, h_mult * h
    x0, x1 = cx - win_w, cx + win_w
    y0, y1 = cy - win_h, cy + win_h
    nearby = []
    for wd in words:
        wcx, wcy = wd["x"] + wd["w"] / 2, wd["y"] + wd["h"] / 2
        if x0 <= wcx <= x1 and y0 <= wcy <= y1:
            nearby.append(wd)
    nearby.sort(key=lambda d: (round(d["y"] / max(1, h)), d["x"]))  # reading order
    return nearby


def _fuzzy_contains(word, keyword, max_dist):
    if word == keyword:
        return True
    if max_dist <= 0:
        return False
    import difflib
    sm = difflib.SequenceMatcher(None, word, keyword)
    matches = sum(b.size for b in sm.get_matching_blocks())
    dist = max(len(word), len(keyword)) - matches
    return dist <= max_dist


def text_flow_and_tag(nearby_words, cfg: Config):
    tokens = [w["text"].strip(":,.-()").upper() for w in nearby_words]
    full_text = " ".join(w["text"] for w in nearby_words)

    found_from = any(any(_fuzzy_contains(t, kw, cfg.fuzzy_max_distance) for kw in cfg.from_keywords)
                      for t in tokens)
    found_to = any(any(_fuzzy_contains(t, kw, cfg.fuzzy_max_distance) for kw in cfg.to_keywords)
                    for t in tokens)

    flow = "unknown"
    if found_from and not found_to:
        flow = "inflow"
    elif found_to and not found_from:
        flow = "outflow"

    tag_match = cfg.tag_pattern.search(full_text)
    tag_ref = tag_match.group(0) if tag_match else None

    return flow, tag_ref, full_text


# --------------------------------------------------------------------------
# combination logic (same shape as v3)
# --------------------------------------------------------------------------
def combine_signals(text_flow, geo_flow, geo_sharpness, cfg: Config):
    if text_flow != "unknown" and geo_flow != "unknown":
        if text_flow == geo_flow:
            return text_flow, "text+geometry (agree)", cfg.agree_confidence
        return text_flow, "text+geometry (CONFLICT - verify)", cfg.conflict_confidence
    if text_flow != "unknown":
        return text_flow, "text-only", cfg.text_only_confidence
    if geo_flow != "unknown":
        conf = cfg.geometry_only_base_confidence + 0.4 * geo_sharpness
        return geo_flow, "geometry-only", round(conf, 2)
    return "unknown", "none", 0.0


def probe_side_has_line(binary_img, bbox, side, cfg: Config, img_w, img_h):
    """Soft diagnostic signal in v4 - not a hard gate (see docstring)."""
    x, y, w, h = bbox
    probe_len = max(8, int(cfg.probe_len_mult * h))
    band = max(2, int(cfg.probe_band_px_mult * h))

    if side in ("left", "right"):
        ymid = y + h // 2
        y0 = max(0, ymid - band // 2)
        y1 = min(binary_img.shape[0], ymid + band // 2 + 1)
        if side == "left":
            x0, x1 = max(0, x - probe_len), x
        else:
            x0, x1 = x + w, min(binary_img.shape[1], x + w + probe_len)
        if x1 <= x0:
            return False
        strip = binary_img[y0:y1, x0:x1]
        col_has_ink = (strip > 0).any(axis=0)
        ordered = col_has_ink[::-1] if side == "left" else col_has_ink
    else:
        xmid = x + w // 2
        x0 = max(0, xmid - band // 2)
        x1 = min(binary_img.shape[1], xmid + band // 2 + 1)
        if side == "top":
            y0, y1 = max(0, y - probe_len), y
        else:
            y0, y1 = y + h, min(binary_img.shape[0], y + h + probe_len)
        if y1 <= y0:
            return False
        strip = binary_img[y0:y1, x0:x1]
        col_has_ink = (strip > 0).any(axis=1)
        ordered = col_has_ink[::-1] if side == "top" else col_has_ink

    if ordered.size == 0:
        return False
    gap_tolerance = 2
    run, gap = 0, 0
    for v in ordered:
        if v:
            run += 1
            gap = 0
        else:
            gap += 1
            if gap > gap_tolerance:
                break
            run += 1
    return (run / len(ordered)) >= cfg.min_line_run_frac


def detect(image_path, cfg: Config = Config()):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = load_binary(gray, cfg)

    words = ocr_words(gray)
    char_h = estimate_char_height(words)

    m_mask = margin_mask(img.shape, cfg)
    excl_mask, excl_rects = build_exclusion_mask(img.shape, words, cfg, w, h)
    valid_mask = m_mask & ~excl_mask

    candidates = find_candidate_shapes(binary, cfg, valid_mask, char_h)
    candidates = dedup_candidates(candidates, cfg.dedup_iou_thresh)

    results = []
    for cand in candidates:
        bbox = cand["bbox"]
        nearby = words_near(words, bbox, cfg.label_search_mult_vertical, cfg.label_search_mult_horizontal)
        nearby_text_upper = " ".join(wd["text"] for wd in nearby).upper()

        if any(kw in nearby_text_upper for kw in cfg.exclude_keywords):
            continue

        text_flow, tag_ref, full_text = text_flow_and_tag(nearby, cfg)

        if cfg.require_tag_pattern_nearby and tag_ref is None:
            continue  # no structured drawing/line reference nearby -> probably not a real connector

        shape_type, tip_side, tip_sharp = classify_shape(binary, cand["contour"], bbox, cfg)
        edge = page_edge(bbox, w, h, cfg)

        if edge in ("left", "right"):
            into_page = "right" if edge == "left" else "left"
            out_of_page = edge
        elif edge in ("top", "bottom"):
            into_page = "down" if edge == "top" else "up"
            out_of_page = "up" if edge == "top" else "down"
        else:
            into_page = out_of_page = "unknown"

        geo_flow = "unknown"
        if shape_type == "chevron":
            geo_flow = geo_flow_from_tip(tip_side, edge, into_page, out_of_page)

        flow, source, confidence = combine_signals(text_flow, geo_flow, tip_sharp, cfg)

        if flow == "inflow":
            arrow_direction = into_page
        elif flow == "outflow":
            arrow_direction = out_of_page
        else:
            arrow_direction = "unknown"

        line_sides = ("left", "right") if edge in ("left", "right", "unknown") else ("top", "bottom")
        has_line = any(probe_side_has_line(binary, bbox, s, cfg, w, h) for s in line_sides)

        results.append({
            "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
            "page_edge": edge,
            "shape_type": shape_type,
            "flow": flow,
            "source": source,
            "text_flow": text_flow,
            "geo_flow": geo_flow,
            "tag_ref": tag_ref,
            "arrow_direction": arrow_direction,
            "confidence": round(float(confidence), 2),
            "has_attached_line": bool(has_line),
            "nearby_text": full_text,
        })

    debug = {
        "char_height_px": char_h,
        "excluded_regions": [{"x0": r[0], "y0": r[1], "x1": r[2], "y1": r[3]} for r in excl_rects],
    }
    return img, results, debug


def annotate(img, results):
    vis = img.copy()
    for r in results:
        x, y, w, h = r["bbox"]["x"], r["bbox"]["y"], r["bbox"]["w"], r["bbox"]["h"]
        if r["flow"] == "unknown":
            color = (0, 0, 255)
        elif "CONFLICT" in r["source"]:
            color = (0, 140, 255)
        else:
            color = (0, 200, 0)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
        label = f'{r["tag_ref"] or "?"} [{r["shape_type"]}/{r["flow"]}/{r["confidence"]}]'
        cv2.putText(vis, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return vis


def parse_rect(s):
    parts = [float(p) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected 4 comma-separated fractions: x0,y0,x1,y1")
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser(description="Detect off-page connectors in a P&ID sheet (v4).")
    ap.add_argument("image", help="path to the P&ID image")
    ap.add_argument("--out-dir", default="offset_detector_output", help="where to write results")
    ap.add_argument("--sides", default="left,right,top,bottom", help="comma list: left,right,top,bottom")
    ap.add_argument("--margin-frac", type=float, default=0.16)
    ap.add_argument("--exclude-rect", action="append", type=parse_rect, default=[],
                     help="x0,y0,x1,y1 as fractions of image size (0-1); repeatable")
    ap.add_argument("--density-thresh", type=int, default=None,
                     help="override auto dense-region word-count threshold")
    ap.add_argument("--no-auto-exclude", action="store_true")
    ap.add_argument("--no-require-tag", action="store_true",
                     help="don't require a nearby drawing/tag reference to accept a candidate "
                          "(use if OCR is too poor on your scans for that gate to be reliable)")
    args = ap.parse_args()

    cfg = Config()
    cfg.search_sides = tuple(args.sides.split(","))
    cfg.edge_margin_frac = args.margin_frac
    cfg.user_exclude_rects = args.exclude_rect
    if args.density_thresh is not None:
        cfg.dense_word_count_thresh = args.density_thresh
    if args.no_auto_exclude:
        cfg.auto_exclude_dense_regions = False
    if args.no_require_tag:
        cfg.require_tag_pattern_nearby = False

    os.makedirs(args.out_dir, exist_ok=True)
    img, results, debug = detect(args.image, cfg)

    image_name = os.path.splitext(os.path.basename(args.image))[0]
    json_path = os.path.join(args.out_dir, f"{image_name}_connectors.json")
    with open(json_path, "w") as f:
        json.dump({"results": results, "debug": debug}, f, indent=2)

    vis = annotate(img, results)
    annotated_path = os.path.join(args.out_dir, f"{image_name}_annotated.png")
    cv2.imwrite(annotated_path, vis)

    n_conflict = sum(1 for r in results if "CONFLICT" in r["source"])
    n_unknown = sum(1 for r in results if r["flow"] == "unknown")
    print(f"char height (calibration) ~ {debug['char_height_px']:.1f}px")
    print(f"auto-excluded {len(debug['excluded_regions'])} dense region(s)")
    print(f"Found {len(results)} candidate off-page connector(s). "
          f"{n_conflict} conflict(s), {n_unknown} unresolved.")
    for r in results:
        print(f'  bbox={r["bbox"]}  shape={r["shape_type"]}  edge={r["page_edge"]}  flow={r["flow"]}  '
              f'source={r["source"]}  tag={r["tag_ref"]}  conf={r["confidence"]}')
    print(f"\nAnnotated image saved to : {annotated_path}")
    print(f"Connector JSON saved to : {json_path}")


if __name__ == "__main__":
    main()