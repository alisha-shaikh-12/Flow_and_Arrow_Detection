"""
Off-page connector detector for P&ID drawings.

An "off-page connector" is the tag block (e.g. the "07A-10123 (14)" box)
that marks where a line leaves one drawing sheet and continues on another.

*** Direction logic, v2 - see note below ***
The first version of this script tried to read flow direction off the
shape of the tag box itself (assuming a pentagon/arrow with a sharp point
on one side). That assumption was wrong: measuring the actual pixel
outline on real sheets shows these boxes have SYMMETRIC chamfered corners
on both ends, not a one-sided arrow. As a result the old logic silently
degenerated into just reporting which page margin the box was closest to
(left box -> "left", right box -> "right"), which is trivially true and
carries no real flow-direction information - that was the bug.

The reliable signal turns out to be textual, not geometric: every one of
these tags has an associated label reading "<fluid> FROM <source>" or
"<fluid> TO <destination>", written just above the box. That FROM/TO
keyword is drafted specifically to convey flow direction, so we OCR that
label and use it directly:
    FROM  -> "inflow"  (flow enters this sheet at this point)
    TO    -> "outflow" (flow leaves this sheet at this point)

We combine that with which page edge the box sits on (left/right/top/
bottom) purely for drawing a sensible arrow in the visualization - e.g. an
"inflow" tag on the left margin gets an arrow pointing rightward, into the
page, since that's the physical direction the fluid is moving.

Usage:
    python detect_offpage_connectors.py input.jpg --out-dir results/
"""

import argparse
import json
import os

import cv2
import numpy as np

try:
    import pytesseract
    HAVE_OCR = True
except ImportError:
    HAVE_OCR = False


# --------------------------------------------------------------------------
# Tunable parameters. All sizes are expressed as fractions of the image
# dimensions so the same numbers work whether you feed in an 8000px sheet
# scan or a smaller export.
# --------------------------------------------------------------------------
class Config:
    # ---- where to look ----
    edge_margin_frac = 0.14      # search this fraction of width/height in from each edge
    search_sides = ("left", "right")   # also allowed: "top", "bottom"

    # ---- binarization ----
    # Circles/thin strokes need a lighter threshold than bold black text;
    # 225-235 works well for scanned blueprint-style line art.
    thresh_value = 228
    close_kernel = 3
    close_iters = 1

    # ---- candidate box filtering ----
    min_width_frac = 0.02        # min box width as fraction of image width
    max_width_frac = 0.14
    min_height_frac = 0.008
    max_height_frac = 0.017      # tuned to admit the ~69px tag box but reject the
                                  # much taller (~110px) revision-table boxes near
                                  # the title block, which have a similar aspect ratio
    min_aspect = 2.0             # width / height
    max_aspect = 8.0

    # keywords that mean "this is a title-block / revision-table box, not a
    # flow-tag connector" - only the drawing-number pattern should survive
    exclude_keywords = ("ISSUED", "BULLETIN", "REVISION", "RECORD", "CONSTRUCTION",
                         "FOR", "CHK", "APPD", "DATE", "CHG", "BY")

    # ---- line-attachment probe (still used to help validate a box is a
    # real connector - a genuine tag has a line attached on exactly one
    # side - but no longer used to infer direction, see module docstring) ----
    probe_len_frac = 0.03        # how far to look outside the box for an attached line
    probe_band_px = 6            # thickness of the horizontal strip probed, in px
    min_line_run_frac = 0.7      # required unbroken run (from the box edge outward)
                                  # to count as a real attached pipe line

    # ---- deduplication ----
    dedup_iou_thresh = 0.4       # merge candidate boxes that overlap more than this
                                  # (thin strokes get traced as inner+outer contours)

    # ---- FROM/TO label search (this is what actually determines direction) ----
    label_search_height_frac = 0.05   # how far above the box to look for the label line
    label_search_width_mult = 3.0     # label text can run wider than the box itself
    from_keywords = ("FROM",)
    to_keywords = ("TO",)


def load_binary(gray, cfg: Config):
    _, th = cv2.threshold(gray, cfg.thresh_value, 255, cv2.THRESH_BINARY_INV)
    if cfg.close_iters > 0:
        k = np.ones((cfg.close_kernel, cfg.close_kernel), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=cfg.close_iters)
    return th


def margin_mask(shape, cfg: Config):
    """Boolean mask that is True only within edge_margin_frac of the chosen sides."""
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


def find_candidate_boxes(binary_img, cfg: Config, img_w, img_h, m_mask):
    # NOTE: RETR_EXTERNAL is a trap here - P&ID sheets have an outer border
    # frame around the whole page, so with EXTERNAL retrieval that frame
    # becomes the one and only "external" contour and everything else
    # (including the connector boxes we actually want) is demoted to an
    # internal/child contour and silently dropped. RETR_LIST returns every
    # contour regardless of nesting, so we filter out the odd giant frame
    # contour ourselves via the width/height bounds below.
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect = w / h
        cx, cy = x + w / 2, y + h / 2

        if not m_mask[int(cy), int(cx)]:
            continue
        if not (cfg.min_width_frac * img_w <= w <= cfg.max_width_frac * img_w):
            continue
        if not (cfg.min_height_frac * img_h <= h <= cfg.max_height_frac * img_h):
            continue
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
    """Thin strokes often get traced as near-identical inner/outer contours;
    collapse overlapping duplicates, keeping the larger of each cluster."""
    candidates = sorted(candidates, key=lambda c: c["bbox"][2] * c["bbox"][3], reverse=True)
    kept = []
    for cand in candidates:
        if all(iou(cand["bbox"], k["bbox"]) < thresh for k in kept):
            kept.append(cand)
    return kept


def probe_side_has_line(binary_img, bbox, side, cfg: Config, img_w):
    """Check whether a real pipe line is physically attached to the box on `side`.

    The key discriminator is NOT "is there any ink somewhere nearby" (that
    picks up unrelated text/symbols sitting near the box) but "does an
    unbroken run of ink start right at the box's edge and continue outward".
    A genuine attached process line touches the box directly; a connector's
    open/pointed side has a gap before the next bit of nearby artwork.
    """
    x, y, w, h = bbox
    ymid = y + h // 2
    y0 = max(0, ymid - cfg.probe_band_px // 2)
    y1 = min(binary_img.shape[0], ymid + cfg.probe_band_px // 2 + 1)

    probe_len = max(10, int(cfg.probe_len_frac * img_w))
    if side == "left":
        x0 = max(0, x - probe_len)
        x1 = x
    else:  # right
        x0 = x + w
        x1 = min(binary_img.shape[1], x + w + probe_len)

    if x1 <= x0:
        return False, 0.0

    strip = binary_img[y0:y1, x0:x1]
    col_has_ink = (strip > 0).any(axis=0)
    if col_has_ink.size == 0:
        return False, 0.0

    # walk outward from the box edge and measure the unbroken leading run
    ordered = col_has_ink[::-1] if side == "left" else col_has_ink
    gap_tolerance = 2  # px, forgives anti-aliasing/JPEG speckle gaps
    run, gap = 0, 0
    for v in ordered:
        if v:
            run += 1
            gap = 0
        else:
            gap += 1
            if gap > gap_tolerance:
                break
            run += 1  # tolerate the small gap itself within the run

    run_frac = run / len(ordered)
    return run_frac >= cfg.min_line_run_frac, run_frac


def page_edge(bbox, img_w, img_h, cfg: Config):
    """Which margin the box was found in - used only to orient the arrow
    in the visualization, not to determine flow direction."""
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    mw, mh = img_w * cfg.edge_margin_frac, img_h * cfg.edge_margin_frac
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


def read_flow_label(gray, bbox, img_w, img_h, cfg: Config):
    """Read the FROM/TO keyword out of the label text drafted above the box.

    This is the actual ground-truth signal for direction on these sheets
    (see module docstring) - the tag's own outline is symmetric and does
    not encode direction, but its label is always written as either
    "<fluid> FROM <source>" or "<fluid> TO <destination>".
    """
    if not HAVE_OCR:
        return "unknown", ""

    x, y, w, h = bbox
    search_h = int(img_h * cfg.label_search_height_frac)
    x0 = max(0, x - 100)
    x1 = min(img_w, x + int(w * cfg.label_search_width_mult))
    y0 = max(0, y - search_h)
    y1 = max(0, y - 5)
    if y1 <= y0 or x1 <= x0:
        return "unknown", ""

    region = gray[y0:y1, x0:x1]
    text = pytesseract.image_to_string(region, config="--psm 6")
    text = " ".join(text.split())
    words = text.upper().split()

    if any(kw in words for kw in cfg.from_keywords):
        return "inflow", text
    if any(kw in words for kw in cfg.to_keywords):
        return "outflow", text
    return "unknown", text


def ocr_text(gray, bbox, pad=2):
    if not HAVE_OCR:
        return ""
    x, y, w, h = bbox
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = x + w + pad, y + h + pad
    crop = gray[y0:y1, x0:x1]
    # upscale small crops; tesseract likes ~300dpi-ish stroke widths
    scale = max(1, 300 // max(1, h))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, crop_bin = cv2.threshold(crop, 200, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(
        crop_bin, config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-()"
    )
    return " ".join(text.split())


def detect(image_path, cfg: Config = Config()):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = load_binary(gray, cfg)
    m_mask = margin_mask(img.shape, cfg)

    candidates = find_candidate_boxes(binary, cfg, w, h, m_mask)
    candidates = dedup_candidates(candidates, cfg.dedup_iou_thresh)

    results = []
    for cand in candidates:
        bbox = cand["bbox"]

        # drop title-block / revision-table boxes before doing anything else
        text_preview = ocr_text(gray, bbox)
        if any(kw in text_preview.upper() for kw in cfg.exclude_keywords):
            continue

        # sanity check: a real connector has a line attached on exactly one
        # side (this just validates the candidate; it no longer decides direction)
        left_line, _ = probe_side_has_line(binary, bbox, "left", cfg, w)
        right_line, _ = probe_side_has_line(binary, bbox, "right", cfg, w)
        has_attached_line = left_line != right_line  # exactly one side, i.e. XOR

        edge = page_edge(bbox, w, h, cfg)
        flow, label_text = read_flow_label(gray, bbox, w, h, cfg)

        # arrow direction for the visualization: inflow points from the
        # margin INTO the page; outflow points from the page OUT to the margin
        if edge in ("left", "right"):
            into_page = "right" if edge == "left" else "left"
            out_of_page = edge
        elif edge in ("top", "bottom"):
            into_page = "down" if edge == "top" else "up"
            out_of_page = "up" if edge == "top" else "down"
        else:
            into_page = out_of_page = "unknown"

        if flow == "inflow":
            arrow_direction = into_page
        elif flow == "outflow":
            arrow_direction = out_of_page
        else:
            arrow_direction = "unknown"

        confidence = 0.9 if flow != "unknown" else 0.0
        if flow != "unknown" and not has_attached_line:
            confidence *= 0.6  # flag it, but don't discard - OCR can still be right

        results.append({
            "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
            "page_edge": edge,
            "flow": flow,               # "inflow" / "outflow" / "unknown"
            "arrow_direction": arrow_direction,
            "confidence": round(float(confidence), 2),
            "text": text_preview,
            "label_text": label_text,
        })

    return img, results


def annotate(img, results):
    vis = img.copy()
    for r in results:
        x, y, w, h = r["bbox"]["x"], r["bbox"]["y"], r["bbox"]["w"], r["bbox"]["h"]
        color = (0, 200, 0) if r["flow"] != "unknown" else (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)

        cx, cy = x + w // 2, y + h // 2
        d = r["arrow_direction"]
        arrow_len = 60
        endpoints = {
            "right": ((x + w, cy), (x + w + arrow_len, cy)),
            "left": ((x, cy), (x - arrow_len, cy)),
            "down": ((cx, y + h), (cx, y + h + arrow_len)),
            "up": ((cx, y), (cx, y - arrow_len)),
        }
        if d in endpoints:
            p0, p1 = endpoints[d]
            cv2.arrowedLine(vis, p0, p1, color, 4, tipLength=0.4)

        label = f'{r["text"] or "?"} [{r["flow"]}]'
        cv2.putText(vis, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return vis


def main():
    ap = argparse.ArgumentParser(description="Detect off-page connectors in a P&ID sheet.")
    ap.add_argument("image", help="path to the P&ID image")
    ap.add_argument("--out-dir", default="output", help="where to write results")
    ap.add_argument("--sides", default="left,right", help="comma list: left,right,top,bottom")
    ap.add_argument("--margin-frac", type=float, default=0.14, help="edge margin as fraction of image size")
    args = ap.parse_args()

    cfg = Config()
    cfg.search_sides = tuple(args.sides.split(","))
    cfg.edge_margin_frac = args.margin_frac

    os.makedirs(args.out_dir, exist_ok=True)
    img, results = detect(args.image, cfg)

    with open(os.path.join(args.out_dir, "connectors.json"), "w") as f:
        json.dump(results, f, indent=2)

    vis = annotate(img, results)
    cv2.imwrite(os.path.join(args.out_dir, "annotated.png"), vis)

    print(f"Found {len(results)} candidate off-page connector(s).")
    for r in results:
        print(f'  bbox={r["bbox"]}  edge={r["page_edge"]}  flow={r["flow"]}  '
              f'arrow={r["arrow_direction"]}  conf={r["confidence"]}  text="{r["text"]}"')


if __name__ == "__main__":
    main()