"""
Off-page connector detector for P&ID drawings (Westlake/Technip drawing family
and generalizable to other flag-style off-page connector symbols).

The symbol is a two-cell rectangular "flag": one cell carries the destination
drawing/line tag (e.g. "F-392"), the other carries the sheet grid reference
letter (e.g. "A-4"), and one end of the rectangle is capped with a triangular
point that indicates flow direction. A "FROM <equipment>" / "TO <equipment>"
caption is printed directly below the symbol.

KEY FIX vs v1: the two cells are drawn as fully separate closed regions (a
complete vertical divider line from top border to bottom border), so
cv2.findContours returns them as two independent contours -- one pentagon
(rectangle + triangular tip) and one plain quadrilateral. v1 only looked at
the pentagon half and therefore lost the second cell's text (usually the
grid-reference letter). v2 explicitly searches for and merges the companion
cell before doing OCR.

Usage:
    python3 detect_connectors_v2.py <input.pdf|input.png> --dpi 300 --out out_dir
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image


# --------------------------------------------------------------------------
# Config: tunable geometry constraints for a connector-flag symbol family.
# Relative-friendly (aspect ratio / vertex count / fraction-of-page) so the
# same config generalizes across drawings rendered at different DPI, as long
# as symbol proportions stay consistent within a drawing family. Different
# P&ID drawing series (e.g. Aker Kvaerner vs Technip/Westlake) can subclass
# this with their own tolerances.
# --------------------------------------------------------------------------
class ConnectorConfig:
    name = "westlake_technip_flag"

    # bounding-box aspect ratio (width / height) of a single flag CELL
    # (the tip-bearing pentagon half only -- companion cell can be narrower)
    min_aspect = 1.2
    max_aspect = 9.0

    # cell height, as a fraction of page height (flag symbols are a fixed
    # template size regardless of drawing content)
    min_height_frac = 0.004
    max_height_frac = 0.02

    # contour area, as a fraction of page area (tip-bearing cell only)
    min_area_frac = 0.00003
    max_area_frac = 0.0009

    # approxPolyDP epsilon as a fraction of contour perimeter
    approx_epsilon_frac = 0.01

    # a valid tip-bearing cell has 5 vertices: 4 rectangle corners + 1 apex
    valid_vertex_counts = (5,)

    # how far below the merged shape's bbox to search for the FROM/TO caption
    label_band_height_frac = 1.8  # multiple of shape bbox height

    # binary threshold for line-art vs background
    bin_threshold = 200

    # companion-cell search tolerances
    companion_max_gap_frac = 0.03      # max x-gap vs page width
    companion_height_tol_frac = 0.35   # allowed relative height mismatch
    companion_y_overlap_min = 0.6      # min fraction of y-range overlap
    companion_min_width_ratio = 0.12   # companion width vs tip-cell width
    companion_max_width_ratio = 1.6


class AkerKvaernerFlagConfig(ConnectorConfig):
    """Placeholder subclass for the Aker Kvaerner connector-flag template.

    Aker Kvaerner drawings in this project tend to render the flag symbol
    slightly larger relative to the sheet and sometimes use a hexagonal
    (6-vertex) cap instead of a triangular (5-vertex) one -- confirm against
    a sample sheet and adjust valid_vertex_counts / aspect / area fractions
    accordingly before use. Kept as a distinct subclass (rather than extra
    branching in the shared functions) so each family's tolerances can be
    tuned independently without risking regressions on the other.
    """
    name = "aker_kvaerner_flag"
    valid_vertex_counts = (5, 6)
    min_aspect = 1.0
    max_aspect = 10.0


def render_pdf_page(pdf_path: Path, dpi: int, out_dir: Path):
    """Rasterize every page of a P&ID PDF to PNG using pdftoppm."""
    prefix = out_dir / "page"
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
        check=True,
    )
    pages = sorted(out_dir.glob("page-*.png"))
    if not pages:
        raise RuntimeError("pdftoppm produced no output pages")
    return pages


def find_all_contours(gray: np.ndarray, cfg: ConnectorConfig):
    _, binary = cv2.threshold(gray, cfg.bin_threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_tip_cells(contours, gray_shape, cfg: ConnectorConfig):
    """Find pentagon (rectangle + triangular tip) contours -- one per connector."""
    h_img, w_img = gray_shape
    page_area = h_img * w_img

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (cfg.min_area_frac * page_area <= area <= cfg.max_area_frac * page_area):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        if not (cfg.min_height_frac * h_img <= h <= cfg.max_height_frac * h_img):
            continue
        aspect = w / float(h)
        if not (cfg.min_aspect <= aspect <= cfg.max_aspect):
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, cfg.approx_epsilon_frac * peri, True).reshape(-1, 2)
        if len(approx) not in cfg.valid_vertex_counts:
            continue

        candidates.append({"bbox": (x, y, w, h), "polygon": approx, "area": area})

    return candidates


def classify_tip_direction(polygon: np.ndarray, bbox_w: int):
    """Determine whether the connector's triangular point faces left or right."""
    xs = polygon[:, 0]
    xmin, xmax = xs.min(), xs.max()
    tol = max(3, int(0.02 * bbox_w))
    left_count = int(np.sum(xs <= xmin + tol))
    right_count = int(np.sum(xs >= xmax - tol))
    if left_count < right_count:
        return "left"
    if right_count < left_count:
        return "right"
    return "unknown"


def rectangle_body_span(polygon: np.ndarray, tip_side: str):
    """Return (x0, x1) of the tip-cell's rectangular part, excluding the tip apex."""
    xs = polygon[:, 0]
    if tip_side == "right":
        x0 = int(xs.min())
        x1 = int(np.sort(xs)[-2])
    elif tip_side == "left":
        x1 = int(xs.max())
        x0 = int(np.sort(xs)[1])
    else:
        x0, x1 = int(xs.min()), int(xs.max())
    return x0, x1


def find_companion_cell(tip_bbox, tip_body_x0, tip_body_x1, tip_side, contours, cfg, page_w):
    """Locate the adjacent plain-rectangle cell sharing the flag's divider edge."""
    tx, ty, tw, th = tip_bbox
    body_w = tip_body_x1 - tip_body_x0
    max_gap = max(4, int(cfg.companion_max_gap_frac * page_w))

    best = None
    best_gap = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        # height similarity
        if abs(h - th) > cfg.companion_height_tol_frac * th:
            continue
        # y overlap
        overlap = max(0, min(y + h, ty + th) - max(y, ty))
        if overlap < cfg.companion_y_overlap_min * min(h, th):
            continue
        # width plausibility
        ratio = w / float(body_w) if body_w else 0
        if not (cfg.companion_min_width_ratio <= ratio <= cfg.companion_max_width_ratio):
            continue

        if tip_side == "right":
            # companion sits to the LEFT of the tip-cell's flat (left) edge
            gap = tip_body_x0 - (x + w)
        elif tip_side == "left":
            # companion sits to the RIGHT of the tip-cell's flat (right) edge
            gap = x - tip_body_x1
        else:
            continue

        if gap < -3 or gap > max_gap:
            continue
        if best_gap is None or abs(gap) < abs(best_gap):
            best = (x, y, w, h)
            best_gap = gap

    return best


def _trim_border_columns(gray_crop, dark_thresh=140, max_trim_frac=0.25):
    """Trim near-solid dark columns at the left/right edges of a cell crop.

    Cell OCR crops are bounded by the connector's own border lines. Even a
    2-3px inward pad can leave a sliver of that border, which tesseract
    regularly misreads as a stray leading/trailing character (e.g. "C" before
    a real "A-4"). This trims full-height dark columns from each edge before
    OCR, up to max_trim_frac of the crop width.
    """
    h, w = gray_crop.shape
    if w < 6:
        return 0, w
    max_trim = max(1, int(w * max_trim_frac))
    col_dark_frac = (gray_crop < dark_thresh).mean(axis=0)
    left = 0
    while left < max_trim and col_dark_frac[left] > 0.6:
        left += 1
    right = w
    while (w - right) < max_trim and col_dark_frac[right - 1] > 0.6:
        right -= 1
    if right <= left:
        return 0, w
    return left, right


def ocr_cell(img, gray, box, upscale=6):
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    if x1 <= x0 or y1 <= y0:
        return "", 0.0
    sub_gray = gray[y0:y1, x0:x1]
    if sub_gray.size == 0:
        return "", 0.0
    left, right = _trim_border_columns(sub_gray)
    x02, x12 = x0 + left, x0 + right
    crop = img.crop((x02, y0, x12, y1))
    if crop.width == 0 or crop.height == 0:
        return "", 0.0
    crop = crop.resize((crop.width * upscale, crop.height * upscale), Image.LANCZOS)
    data = pytesseract.image_to_data(
        crop, output_type=pytesseract.Output.DICT,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
    )
    words = [w for w in data["text"] if w.strip()]
    confs = [float(c) for c, w in zip(data["conf"], data["text"]) if w.strip()]
    text = "".join(words).strip()
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return text, avg_conf


LABEL_RE = re.compile(r"\b(FROM|TO)\b[\s:]*([A-Z]{1,6}[\-\s]?\d{3,5}[A-Z]?)", re.IGNORECASE)


def ocr_label_band(img: Image.Image, bbox, cfg: ConnectorConfig, page_size, upscale=3):
    x, y, w, h = bbox
    band_h = int(h * cfg.label_band_height_frac)
    pad_x = int(w * 0.15)
    x0 = max(0, x - pad_x)
    x1 = min(page_size[0], x + w + pad_x)
    y0 = y + h
    y1 = min(page_size[1], y0 + band_h)
    crop = img.crop((x0, y0, x1, y1))
    if crop.width == 0 or crop.height == 0:
        return None, ""
    crop = crop.resize((crop.width * upscale, crop.height * upscale), Image.LANCZOS)
    raw = pytesseract.image_to_string(crop, config="--psm 6").strip()
    cleaned = raw.replace("\n", " ")
    m = LABEL_RE.search(cleaned)
    if not m:
        return None, raw
    direction_word = m.group(1).upper()
    tag = m.group(2).upper().replace(" ", "")
    if "-" not in tag:
        mm = re.match(r"([A-Z]+)(\d+)", tag)
        if mm:
            tag = f"{mm.group(1)}-{mm.group(2)}"
    return {"direction_word": direction_word, "equipment_tag": tag}, raw


def derive_line_prefix(drawing_number: str):
    """Derive the drawing-series letter prefix used by sister-sheet references.

    e.g. "110-PRE-F-391" -> the connectors on this sheet reference sister
    sheets as short forms like "F-392", "F-390" (drop the "110-PRE-" project
    prefix, keep the letter-dash-number tail pattern). Returns the leading
    letter block, e.g. "F", or None if it can't be parsed.
    """
    parts = [p for p in re.split(r"[-_]", drawing_number) if p]
    for i in range(len(parts) - 1, 0, -1):
        if parts[i].isdigit() and parts[i - 1].isalpha():
            return parts[i - 1].upper()
    return None


def classify_cell_semantics(text_a, text_b, line_prefix):
    """Decide which OCR'd cell is the sister-drawing/line number vs the
    sheet grid-reference, using the drawing's own numbering pattern rather
    than tip-adjacency position (position varies by symbol sub-convention).

    Returns (target_drawing_number, sheet_grid_reference, strict_match) where
    strict_match is True only if a cell matched the drawing's OWN letter
    prefix (e.g. "F-" on a drawing whose own number is "...-F-391") -- this
    is used downstream to reject shape-only false positives (equipment tags,
    valve labels, etc. that happen to pass the pentagon geometry test but
    don't reference a sister sheet in this drawing's numbering scheme).
    """
    pat = re.compile(rf"^{re.escape(line_prefix)}-?\d{{3,4}}$") if line_prefix else re.compile(r"^[A-Z]{1,2}-?\d{3,4}$")
    a_match = bool(text_a) and bool(pat.match(text_a))
    b_match = bool(text_b) and bool(pat.match(text_b))
    if a_match and not b_match:
        return text_a, text_b, True
    if b_match and not a_match:
        return text_b, text_a, True
    # fall back: generic "letter-dash-digits" cell with the MOST digits is
    # usually the sister-drawing number; shorter alnum tag is the grid ref
    generic = re.compile(r"^[A-Z]{0,2}-?\d{2,5}$")
    a_generic = bool(text_a) and bool(generic.match(text_a))
    b_generic = bool(text_b) and bool(generic.match(text_b))
    if a_generic and b_generic:
        result = (text_a, text_b) if len(text_a) >= len(text_b) else (text_b, text_a)
        return result[0], result[1], False
    return text_a, text_b, False


def build_connector_record(idx, tip_cand, companion, gray, img, cfg, page_size,
                            drawing_number, source_file, page_num, line_prefix):
    tx, ty, tw, th = tip_cand["bbox"]
    polygon = tip_cand["polygon"]
    tip_side = classify_tip_direction(polygon, tw)
    tip_body_x0, tip_body_x1 = rectangle_body_span(polygon, tip_side)

    flags = []

    def _padded(box):
        x0, y0, x1, y1 = box
        pad_x = min(12, max(2, int((x1 - x0) * 0.12)))
        return (x0 + pad_x, y0 + 3, x1 - pad_x, y1 - 3)

    if companion is None:
        merged_x0, merged_y0 = tx, ty
        merged_x1, merged_y1 = tx + tw, ty + th
        tip_text, tip_conf = ocr_cell(img, gray, _padded((tip_body_x0, ty, tip_body_x1, ty + th)))
        companion_text, companion_conf = "", 0.0
        divider_x = None
        flags.append("companion_cell_not_found")
    else:
        cx, cy, cw, ch = companion
        merged_x0 = min(tx, cx)
        merged_y0 = min(ty, cy)
        merged_x1 = max(tx + tw, cx + cw)
        merged_y1 = max(ty + th, cy + ch)
        tip_text, tip_conf = ocr_cell(img, gray, _padded((tip_body_x0, ty, tip_body_x1, ty + th)))
        companion_text, companion_conf = ocr_cell(img, gray, _padded((cx, cy, cx + cw, cy + ch)))
        divider_x = (tip_body_x0 + (cx + cw)) // 2 if tip_side == "right" else (tip_body_x1 + cx) // 2

    merged_bbox = (merged_x0, merged_y0, merged_x1 - merged_x0, merged_y1 - merged_y0)
    label, label_raw = ocr_label_band(img, merged_bbox, cfg, page_size)

    label_direction = None
    connected_equipment = None
    if label:
        label_direction = "incoming" if label["direction_word"] == "FROM" else "outgoing"
        connected_equipment = label["equipment_tag"]

    # Shape-based direction is convention-dependent (tip can point toward or
    # away from the sheet edge depending on drafting standard) and is only
    # used as a secondary cross-check / fallback when the caption can't be
    # OCR'd -- the caption text is authoritative when present.
    shape_direction = "incoming" if tip_side == "left" else ("outgoing" if tip_side == "right" else None)
    direction = label_direction if label_direction is not None else shape_direction
    direction_agrees = (label_direction is None) or (shape_direction is None) or (label_direction == shape_direction)

    target_drawing_number, sheet_grid_reference, strict_semantic_match = classify_cell_semantics(
        tip_text, companion_text, line_prefix
    )
    # keep OCR confidence attached to whichever raw cell each value came from
    conf_map = {tip_text: tip_conf, companion_text: companion_conf}
    target_conf = conf_map.get(target_drawing_number, 0.0)
    grid_conf = conf_map.get(sheet_grid_reference, 0.0)

    x0, y0, w0, h0 = merged_bbox
    record = {
        "connector_id": f"conn_{page_num:02d}_{idx:04d}",
        "type": "off_page_connector",
        "symbol_family": cfg.name,
        "drawing_number": drawing_number,
        "source_file": source_file,
        "page": page_num,
        "geometry": {
            "bbox_px": [int(x0), int(y0), int(x0 + w0), int(y0 + h0)],
            "tip_polygon_px": polygon.tolist(),
            "companion_bbox_px": (
                [int(companion[0]), int(companion[1]),
                 int(companion[0] + companion[2]), int(companion[1] + companion[3])]
                if companion else None
            ),
            "tip_side": tip_side,
            "divider_x_px": int(divider_x) if divider_x is not None else None,
        },
        "direction": direction,
        "direction_source": "caption" if label_direction is not None else ("shape" if shape_direction is not None else None),
        "target_drawing_number": target_drawing_number or None,
        "target_drawing_number_matches_drawing_series": strict_semantic_match,
        "target_drawing_number_ocr_confidence": round(target_conf, 1),
        "sheet_grid_reference": sheet_grid_reference or None,
        "sheet_grid_reference_ocr_confidence": round(grid_conf, 1),
        "connected_equipment": connected_equipment,
        "label_caption_raw": label_raw.strip() or None,
        "direction_signals_agree": direction_agrees,
        "graph_edge": {
            "from_node": (
                f"{target_drawing_number}:{connected_equipment}" if direction == "incoming" and target_drawing_number
                else f"{drawing_number}:conn_{page_num:02d}_{idx:04d}"
            ),
            "to_node": (
                f"{drawing_number}:conn_{page_num:02d}_{idx:04d}" if direction == "incoming"
                else f"{target_drawing_number}:{connected_equipment}" if target_drawing_number else None
            ),
            "relationship": "flow_continues_to",
        },
        "flags": flags,
    }

    if not target_drawing_number:
        record["flags"].append("target_drawing_number_ocr_empty")
    if not sheet_grid_reference:
        record["flags"].append("sheet_grid_reference_ocr_empty")
    if not connected_equipment:
        record["flags"].append("from_to_caption_not_parsed")
    if not direction_agrees:
        record["flags"].append("shape_and_caption_direction_mismatch")
    if min(target_conf, grid_conf) < 60:
        record["flags"].append("low_ocr_confidence_review_recommended")

    return record


def dedupe_tip_cells(candidates):
    """Guard against the same symbol being matched twice (nested contours)."""
    candidates = sorted(candidates, key=lambda c: -c["area"])
    kept = []
    for c in candidates:
        x, y, w, h = c["bbox"]
        cx, cy = x + w / 2, y + h / 2
        dup = False
        for k in kept:
            kx, ky, kw, kh = k["bbox"]
            if kx <= cx <= kx + kw and ky <= cy <= ky + kh:
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


def annotate(img: Image.Image, records, out_path):
    import PIL.ImageDraw as ImageDraw

    canvas = img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for r in records:
        x0, y0, x1, y1 = r["geometry"]["bbox_px"]
        color = (0, 170, 0) if r["direction"] == "incoming" else (200, 0, 0)
        draw.rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6], outline=color, width=6)
        label = f'{r["connector_id"]} {r["target_drawing_number"]}/{r["sheet_grid_reference"]} {r["direction"]}'
        draw.text((x0, y1 + 8), label, fill=color)
    canvas.save(out_path)


def _merged_bbox_overlap_dedupe(records):
    """Drop duplicate detections whose merged bbox is (near-)contained inside
    another record's merged bbox -- happens when a companion cell's own
    contour also happened to pass the tip-cell (pentagon) test."""

    def area(b):
        x0, y0, x1, y1 = b
        return max(0, x1 - x0) * max(0, y1 - y0)

    recs = sorted(records, key=lambda r: -area(r["geometry"]["bbox_px"]))
    kept = []
    for r in recs:
        x0, y0, x1, y1 = r["geometry"]["bbox_px"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        dup = False
        for k in kept:
            kx0, ky0, kx1, ky1 = k["geometry"]["bbox_px"]
            if kx0 <= cx <= kx1 and ky0 <= cy <= ky1:
                dup = True
                break
        if not dup:
            kept.append(r)
    return kept


def _looks_like_real_connector(rec):
    """Filter shape-only false positives (e.g. valve/instrument tag boxes
    that happen to match the pentagon geometry test): a genuine off-page
    connector must have EITHER a parsed FROM/TO caption OR a cell matching
    the drawing's own sister-sheet numbering pattern."""
    if rec["connected_equipment"]:
        return True
    if rec["target_drawing_number"] and rec.get("target_drawing_number_matches_drawing_series"):
        return True
    return False


def process_page(page_png: Path, page_num: int, cfg, drawing_number, source_file, out_dir, line_prefix):
    img_rgb = Image.open(page_png)
    gray = np.array(img_rgb.convert("L"))

    contours = find_all_contours(gray, cfg)
    tip_cells = find_tip_cells(contours, gray.shape, cfg)
    tip_cells = dedupe_tip_cells(tip_cells)

    page_w = gray.shape[1]

    records = []
    for i, cand in enumerate(
        sorted(tip_cells, key=lambda c: (c["bbox"][1], c["bbox"][0])), start=1
    ):
        tip_side = classify_tip_direction(cand["polygon"], cand["bbox"][2])
        body_x0, body_x1 = rectangle_body_span(cand["polygon"], tip_side)
        companion = find_companion_cell(
            cand["bbox"], body_x0, body_x1, tip_side, contours, cfg, page_w
        )
        rec = build_connector_record(
            i, cand, companion, gray, img_rgb, cfg, gray.shape[::-1],
            drawing_number, source_file, page_num, line_prefix,
        )
        records.append(rec)

    records = _merged_bbox_overlap_dedupe(records)

    kept, dropped = [], []
    for r in records:
        (kept if _looks_like_real_connector(r) else dropped).append(r)
    for r in dropped:
        r["flags"].append("dropped_low_confidence_shape_only_match")

    # renumber sequentially after dedupe for clean, stable IDs
    kept = sorted(kept, key=lambda r: (r["geometry"]["bbox_px"][1], r["geometry"]["bbox_px"][0]))
    for i, r in enumerate(kept, start=1):
        r["connector_id"] = f"conn_{page_num:02d}_{i:04d}"
        r["graph_edge"]["from_node"] = r["graph_edge"]["from_node"].split(":conn_")[0] + f":{r['connector_id']}" \
            if ":conn_" in r["graph_edge"]["from_node"] else r["graph_edge"]["from_node"]
        r["graph_edge"]["to_node"] = r["graph_edge"]["to_node"].split(":conn_")[0] + f":{r['connector_id']}" \
            if r["graph_edge"]["to_node"] and ":conn_" in r["graph_edge"]["to_node"] else r["graph_edge"]["to_node"]

    annotate(img_rgb, kept, out_dir / f"connectors_annotated_p{page_num}.png")
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to PDF or rasterized page image")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default="connector_output")
    ap.add_argument("--drawing-number", default=None)
    ap.add_argument("--family", default="westlake_technip", choices=["westlake_technip", "aker_kvaerner"],
                     help="Drawing-family template to use for symbol geometry tolerances")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == ".pdf":
        pages = render_pdf_page(in_path, args.dpi, out_dir)
    else:
        pages = [in_path]

    cfg = AkerKvaernerFlagConfig() if args.family == "aker_kvaerner" else ConnectorConfig()
    drawing_number = args.drawing_number or in_path.stem
    line_prefix = derive_line_prefix(drawing_number)

    all_records, all_dropped = [], []
    for pnum, page_png in enumerate(pages, start=1):
        kept, dropped = process_page(page_png, pnum, cfg, drawing_number, str(in_path.name), out_dir, line_prefix)
        all_records.extend(kept)
        all_dropped.extend(dropped)

    gray0 = np.array(Image.open(pages[0]).convert("L"))
    out_json = out_dir / "connectors.json"
    payload = {
        "drawing_number": drawing_number,
        "source_file": str(in_path.name),
        "render_dpi": args.dpi,
        "page_size_px": [int(gray0.shape[1]), int(gray0.shape[0])],
        "num_pages": len(pages),
        "line_number_prefix_used_for_semantics": line_prefix,
        "num_connectors_detected": len(all_records),
        "connectors": all_records,
        "num_candidates_rejected_as_non_connectors": len(all_dropped),
        "rejected_candidates_for_qa": all_dropped,
    }
    out_json.write_text(json.dumps(payload, indent=2))

    print(f"Detected {len(all_records)} connectors across {len(pages)} page(s)")
    print(f"Rejected {len(all_dropped)} shape-only false-positive candidates")
    print(f"JSON  -> {out_json}")


if __name__ == "__main__":
    sys.exit(main())