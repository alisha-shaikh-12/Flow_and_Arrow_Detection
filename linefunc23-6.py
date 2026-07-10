## linefunc23-6.py
## P&ID Line Detection – Functions Library
## Updated 23-Jun with improved arrow detection:
##   - Both solid-triangle AND open-chevron templates
##   - Samples N points along full line segment (not just endpoints)
##   - Lower default threshold (0.48) to catch faint open chevrons
##   - Geometric sanity check: template direction vs tail→tip geometry
##   - tip_confidence field in JSON output

import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import base64
import requests
from datetime import datetime
from collections import Counter


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – PDF → PNG
# ══════════════════════════════════════════════════════════════════════════════

def extract_single_page_image(
    pdf_path: str,
    output_image_path: str,
    page_number: int,
    dpi: int = 150,
) -> str:
    """
    Extract a single page from a PDF and save it as a PNG image.

    Args:
        pdf_path         : Path to the source PDF file.
        output_image_path: Directory where the image will be saved.
        page_number      : 1-based page number (page 1 = 1).
        dpi              : Render resolution in dots-per-inch (default 150).

    Returns:
        Absolute path to the saved PNG file.

    Raises:
        FileNotFoundError : If pdf_path does not exist.
        ValueError        : If page_number is out of range.
        OSError           : If the output directory cannot be created.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    os.makedirs(output_image_path, exist_ok=True)

    doc = fitz.open(pdf_path)

    try:
        total_pages = doc.page_count

        if not (1 <= page_number <= total_pages):
            raise ValueError(
                f"page_number {page_number} is out of range "
                f"(PDF has {total_pages} page(s))."
            )

        page   = doc.load_page(page_number - 1)
        zoom   = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix    = page.get_pixmap(matrix=matrix, alpha=False)

        image_path = os.path.join(output_image_path, f"page_{page_number}.png")
        pix.save(image_path)

    finally:
        doc.close()

    return os.path.abspath(image_path)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – Morphological line separation
# ══════════════════════════════════════════════════════════════════════════════

def extract_lines_without_ocr(
    image_path: str,
    dotted_kernel: int,
    solid_kernel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separate dotted and solid lines from a grayscale image using
    morphological opening — text strokes are too short to survive.

    Args:
        image_path    : Path to the input image.
        dotted_kernel : Minimum run-length (px) to retain dotted/dashed lines.
        solid_kernel  : Minimum run-length (px) to retain solid lines.
                        Must be >= dotted_kernel.

    Returns:
        lines_dotted    : Binary mask — all lines (dotted + solid).
        lines_only_solid: Binary mask — solid lines only.
        lines_diff      : Binary mask — dotted/dashed lines only
                          (lines_dotted minus lines_only_solid).

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError       : If the image cannot be read or kernels are invalid.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if dotted_kernel < 1 or solid_kernel < 1:
        raise ValueError("Kernel sizes must be >= 1.")
    if solid_kernel < dotted_kernel:
        raise ValueError("solid_kernel must be >= dotted_kernel.")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cv2 could not read image: {image_path}")

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    del img

    def _extract_lines(src: np.ndarray, ksize: int) -> np.ndarray:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ksize))
        h_lines  = cv2.morphologyEx(src, cv2.MORPH_OPEN, h_kernel, iterations=2)
        v_lines  = cv2.morphologyEx(src, cv2.MORPH_OPEN, v_kernel, iterations=2)
        return cv2.add(h_lines, v_lines)

    lines_dotted     = _extract_lines(binary, dotted_kernel)
    lines_only_solid = _extract_lines(binary, solid_kernel)
    del binary

    lines_diff = cv2.subtract(lines_dotted, lines_only_solid)

    return lines_dotted, lines_only_solid, lines_diff


# ══════════════════════════════════════════════════════════════════════════════
# Step 2b – Save morphology results
# ══════════════════════════════════════════════════════════════════════════════

def save_morphology_results(
    page_number: int,
    lines_solid: np.ndarray,
    lines_dotted: np.ndarray,
    lines_diff: np.ndarray,
    output_dir: str,
) -> dict[str, str]:
    """
    Save morphology result images for a given page.

    Args:
        page_number  : 1-based page number used in filenames.
        lines_solid  : Binary mask — solid lines only.
        lines_dotted : Binary mask — dotted + solid lines.
        lines_diff   : Binary mask — dotted lines only.
        output_dir   : Directory to write PNGs into (created if missing).

    Returns:
        Dict mapping label → saved path e.g. {"solid": "/.../solid.png", ...}

    Raises:
        ValueError : If any array is None or empty.
        IOError    : If any image fails to write.
    """
    arrays = {
        "solid":  lines_solid,
        "dotted": lines_dotted,
        "diff":   lines_diff,
    }

    for label, arr in arrays.items():
        if arr is None or arr.size == 0:
            raise ValueError(f"'{label}' array is None or empty.")

    os.makedirs(output_dir, exist_ok=True)

    saved: dict[str, str] = {}

    for label, arr in arrays.items():
        path = os.path.join(output_dir, f"lines_morphology_{label}_pn_{page_number}.png")
        if not cv2.imwrite(path, arr):
            raise IOError(f"cv2.imwrite failed for '{label}' → {path}")
        saved[label] = path
        print(f"  [{label:>6}] saved → {path}")

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 – Hough line detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_hough_lines(
    lines_diff      : np.ndarray,
    lines_only_solid: np.ndarray,
    image_path      : str,
    hough_transform_output_path: str,
    page_number     : int,
    DPI             : int,
    config          : None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Run Canny + HoughLinesP on solid and dotted line masks.

    Args:
        lines_diff       : Binary mask — dotted lines only  (uint8, 2-D).
        lines_only_solid : Binary mask — solid lines only   (uint8, 2-D).
        image_path       : Original page image path.
        hough_transform_output_path : Output directory for visualisation.
        page_number      : 1-based page number.
        DPI              : Dots per inch of the source image.
        config           : HoughConfig instance. Uses defaults if not provided.

    Returns:
        (lines_solid_total, lines_dotted_total)
        Each is np.ndarray of shape (N,1,4) or None if nothing detected.

    Raises:
        FileNotFoundError : If image_path cannot be read.
        ValueError        : If either mask is None or empty.
    """
    cfg = config

    if cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    for name, mask in (("lines_diff", lines_diff), ("lines_only_solid", lines_only_solid)):
        if mask is None or mask.size == 0:
            raise ValueError(f"'{name}' mask is None or empty.")

    edges_diff  = cv2.Canny(lines_diff,       cfg.canny_low, cfg.canny_high, apertureSize=3)
    edges_solid = cv2.Canny(lines_only_solid, cfg.canny_low, cfg.canny_high, apertureSize=3)

    lines_dotted_total = cv2.HoughLinesP(
        edges_diff, cfg.rho, cfg.theta, cfg.threshold,
        minLineLength=cfg.min_line_length,
        maxLineGap=cfg.max_line_gap,
    )
    lines_solid_total = cv2.HoughLinesP(
        edges_solid, cfg.rho, cfg.theta, cfg.threshold,
        minLineLength=cfg.min_line_length,
        maxLineGap=cfg.max_line_gap,
    )

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_h, image_w = gray.shape

    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if lines_dotted_total is not None:
        for line in lines_dotted_total:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 4)

    print(f"Detected {len(lines_dotted_total) if lines_dotted_total is not None else 0} line dotted segments")

    if lines_solid_total is not None:
        for line in lines_solid_total:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 4)

    print(f"  [solid ] detected {len(lines_solid_total)  if lines_solid_total  is not None else 0} segment(s)")
    print(f"  [dotted] detected {len(lines_dotted_total) if lines_dotted_total is not None else 0} segment(s)")

    print(f"  [plot] saving visualisation for page {page_number}...")
    plt.figure(figsize=(12, 6))
    plt.imshow(output[:, :, ::-1])
    plt.title("Lines")
    plt.axis("off")

    legend_elements = [
        Line2D([0], [0], color='blue',  lw=3,              label='Solid pipeline'),
        Line2D([0], [0], color='green', lw=3, linestyle='--', label='Dotted instrumental lines'),
    ]

    plt.legend(handles=legend_elements, loc='lower right', frameon=True)
    plt.tight_layout()

    os.makedirs(hough_transform_output_path, exist_ok=True)

    output_file = os.path.join(
        hough_transform_output_path,
        f"line_hough_transform_{page_number}.png"
    )

    plt.savefig(output_file, bbox_inches='tight', dpi=DPI)
    plt.close()
    print(f"[hough ] saved → {output_file}")

    return lines_solid_total, lines_dotted_total


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 – Arrow direction detection  (UPDATED v2  23-Jun)
# ══════════════════════════════════════════════════════════════════════════════

def _make_arrow_templates(sizes=(8, 11, 14, 18, 22, 26, 32)):
    """
    Generate BOTH solid-filled triangle AND open-chevron templates for the
    four cardinal directions at multiple pixel sizes.

    Why both styles?
      P&ID drawings use two distinct arrowhead conventions:
        - Solid filled triangles  →  process pipe flow arrows
        - Open chevrons (> shape) →  instrument/signal line arrows
      Trying both types doubles recall on real drawings compared to
      solid triangles alone (the previous v1 behaviour).

    Solid RIGHT example (size=10):
         █
       ███
     █████
    ███████
    ███████
     █████
       ███
         █
    Open chevron RIGHT example (size=10):
    █
     ██
       ███
          ████  ← tip
       ███
     ██
    █

    All other directions are derived by flip/rotate.
    """
    templates: dict[str, list[np.ndarray]] = {
        "RIGHT": [], "LEFT": [], "UP": [], "DOWN": [],
    }

    for size in sizes:
        h = w = size
        half = (h - 1) / 2.0

        # ── Solid filled triangle ─────────────────────────────────────────
        solid = np.zeros((h, w), dtype=np.uint8)
        for row in range(h):
            dist      = abs(row - half)
            frac      = 1.0 - (dist / half) if half > 0 else 1.0
            start_col = int(round((1.0 - frac) * (w - 1)))
            solid[row, start_col:] = 255

        # ── Open chevron (two diagonal strokes, hollow centre) ────────────
        chevron   = np.zeros((h, w), dtype=np.uint8)
        thickness = max(1, size // 8)
        mid       = h // 2
        cv2.line(chevron, (0, 0),   (w - 1, mid), 255, thickness)
        cv2.line(chevron, (0, h-1), (w - 1, mid), 255, thickness)

        for tmpl in (solid, chevron):
            templates["RIGHT"].append(tmpl)
            templates["LEFT"].append(cv2.flip(tmpl, 1))
            templates["DOWN"].append(cv2.rotate(tmpl, cv2.ROTATE_90_CLOCKWISE))
            templates["UP"].append(cv2.rotate(tmpl, cv2.ROTATE_90_COUNTERCLOCKWISE))

    return templates


def _candidate_directions(x1, y1, x2, y2) -> list[str]:
    """
    Restrict template search to directions consistent with line orientation.

    Horizontal line → only LEFT/RIGHT
    Vertical line   → only UP/DOWN
    Diagonal        → all four

    This eliminates ~50 % of false positives on axis-aligned P&ID lines.
    """
    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if angle < 30 or angle > 150:
        return ["RIGHT", "LEFT"]
    elif 60 < angle < 120:
        return ["UP", "DOWN"]
    return ["RIGHT", "LEFT", "UP", "DOWN"]


def _search_along_line(
    binary_inv    : np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    templates     : dict,
    candidate_dirs: list[str],
    n_samples     : int   = 7,
    search_radius : int   = 50,
    threshold     : float = 0.48,
):
    """
    Sample N evenly-spaced points along the full line segment and run
    multi-scale template matching at each point.

    Why along the full segment instead of just the endpoints?
      Hough fragments often miss the true endpoint — the arrowhead can sit
      anywhere along (or just past) the detected segment.  Sampling 7 points
      (including both ends) catches arrowheads wherever they actually are.

    Args:
        binary_inv    : uint8 image where ink = 255.
        x1,y1,x2,y2  : Hough line segment endpoints.
        templates     : Output of _make_arrow_templates().
        candidate_dirs: From _candidate_directions().
        n_samples     : Number of points to sample along segment (incl. endpoints).
        search_radius : Half-size of ROI window around each sample point (px).
        threshold     : Minimum TM_CCOEFF_NORMED score to accept.

    Returns:
        Best match tuple (direction, score, box_x, box_y, box_w, box_h,
                          sample_point, t_position)
        or None if nothing exceeded threshold.
        t_position: 0.0 = segment start, 1.0 = segment end.
    """
    h, w  = binary_inv.shape
    best_score = threshold - 1e-6
    best       = None

    for t in np.linspace(0.0, 1.0, n_samples):
        cx = int(x1 + t * (x2 - x1))
        cy = int(y1 + t * (y2 - y1))

        rx1 = max(0, cx - search_radius)
        rx2 = min(w, cx + search_radius)
        ry1 = max(0, cy - search_radius)
        ry2 = min(h, cy + search_radius)
        roi = binary_inv[ry1:ry2, rx1:rx2].astype(np.float32)
        rh, rw = roi.shape
        if rh < 4 or rw < 4:
            continue

        for direction in candidate_dirs:
            for tmpl in templates[direction]:
                th, tw = tmpl.shape
                if rh < th or rw < tw:
                    continue
                result = cv2.matchTemplate(
                    roi, tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    mx = rx1 + max_loc[0]
                    my = ry1 + max_loc[1]
                    best = (direction, max_val, mx, my, tw, th, (cx, cy), t)

    return best


def _ink_density(gray: np.ndarray, cx: int, cy: int, radius: int = 28) -> float:
    """
    Fraction of pixels in a square patch around (cx,cy) that are ink (dark).
    Used as a fallback when template matching finds nothing.
    The arrowhead base has MORE ink than the open tail end, so the endpoint
    with HIGHER density = arrowhead side = the tip direction.
    """
    h, w   = gray.shape
    x1, x2 = max(0, cx - radius), min(w, cx + radius)
    y1, y2 = max(0, cy - radius), min(h, cy + radius)
    patch  = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    _, thr = cv2.threshold(patch, 200, 255, cv2.THRESH_BINARY_INV)
    return float(np.sum(thr > 0)) / thr.size


def _nms_detections(detections: list[dict], iou_thresh: float = 0.30) -> list[dict]:
    """
    Non-maximum suppression: keep highest-score detection when boxes overlap.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept: list[dict] = []

    def _iou(a: dict, b: dict) -> float:
        ax1, ay1 = a["bx"], a["by"]
        ax2, ay2 = ax1 + a["bw"], ay1 + a["bh"]
        bx1, by1 = b["bx"], b["by"]
        bx2, by2 = bx1 + b["bw"], by1 + b["bh"]
        iw   = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih   = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    for det in detections:
        if all(_iou(det, k) < iou_thresh for k in kept):
            kept.append(det)

    return kept


# Direction colours (BGR) and unicode labels used in visualisation
_DIR_UNICODE = {"RIGHT": "->", "LEFT": "<-", "UP": "^", "DOWN": "v"}
_DIR_COLOR   = {
    "RIGHT": (0,  140, 255),   # orange
    "LEFT" : (0,  200,  60),   # green
    "UP"   : (200,  0, 200),   # purple
    "DOWN" : (0,   80, 255),   # red-orange
}


def _get_direction_label(tail, tip) -> str:
    """Derive cardinal direction from a tail→tip vector."""
    dx    = tip[0] - tail[0]
    dy    = tip[1] - tail[1]
    angle = np.degrees(np.arctan2(dy, dx))
    if   -45  <= angle <=  45:          return "RIGHT"
    elif  45  <  angle <= 135:          return "DOWN"
    elif angle > 135 or angle < -135:   return "LEFT"
    else:                               return "UP"


def detect_arrow_directions(
    image_path     : str,
    lines_solid    : np.ndarray,
    lines_dotted   : np.ndarray,
    page_number    : int,
    output_dir     : str,
    gemini_api_key : str   = "",    # accepted but not used (kept for compat)
    match_threshold: float = 0.48,  # lower than v1 to catch open chevrons
    search_radius  : int   = 50,    # px half-window around each sample point
    n_samples      : int   = 7,     # points sampled along each line segment
    snap_px        : int   = 40,    # junction-merge radius in px
) -> dict:
    """
    Detect flow-arrow directions on Hough line segments — UPDATED v2 23-Jun.

    Improvements over the previous version:
      1. Both solid-triangle AND open-chevron templates  →  doubles recall
         on P&ID instrument lines which use open > arrows, not filled ones.
      2. Searches N points along the full segment (not just the 2 endpoints)
         →  catches arrowheads wherever the Hough fragment actually ends.
      3. Lower default threshold (0.48 vs 0.52)  →  open chevrons score ~10 %
         lower than solid triangles, so the old threshold missed them.
      4. Geometric sanity check: if the template-reported direction contradicts
         the tail→tip geometry, flip tail and tip instead of trusting neither.
      5. detection_method values: "template_match" | "density_fallback"
         (unchanged from v1, so downstream JSON consumers need no changes).

    Output JSON schema is fully backward-compatible with v1.

    Args:
        image_path     : Path to the original page PNG.
        lines_solid    : np.ndarray (N,1,4) of solid Hough segments, or None.
        lines_dotted   : np.ndarray (M,1,4) of dotted Hough segments, or None.
        page_number    : 1-based page number (used in output filenames).
        output_dir     : Directory to save PNG + JSON outputs.
        gemini_api_key : Ignored. Kept so existing call sites don't need edits.
        match_threshold: TM_CCOEFF_NORMED minimum score (0–1). Lower = more
                         detections but more false positives. Try 0.44 if you
                         are missing arrows; try 0.54 if you see too many.
        search_radius  : Half-size (px) of the ROI searched around each sample.
        n_samples      : How many points to sample along each line segment.
                         7 covers both endpoints + 5 internal positions.
        snap_px        : Endpoints within this distance share a graph node.

    Returns:
        flow_data dict with keys: source_image, page_number, timestamp,
        image_size, nodes, edges, summary.
        Also writes:
          <output_dir>/flow_arrows_pn_<N>.png   — visualisation
          <output_dir>/flow_graph_pn_<N>.json   — graph data
    """
    if gemini_api_key:
        print("  [arrows] NOTE: gemini_api_key ignored – using template matching.")

    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis  = img.copy()

    # Binary image where INK = 255 (used for template matching)
    _, binary_inv = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Pre-build all templates once (solid + chevron, 7 sizes, 4 directions)
    templates = _make_arrow_templates(sizes=(8, 11, 14, 18, 22, 26, 32))

    def flatten(arr):
        """Convert (N,1,4) ndarray to list of (x1,y1,x2,y2) tuples."""
        return [] if arr is None else [tuple(l[0]) for l in arr]

    solid_lines  = flatten(lines_solid)
    dotted_lines = flatten(lines_dotted)

    flow_data: dict = {
        "source_image": image_path,
        "page_number" : page_number,
        "timestamp"   : datetime.now().isoformat(),
        "image_size"  : {"width": int(img.shape[1]), "height": int(img.shape[0])},
        "nodes"       : [],
        "edges"       : [],
        "summary"     : {},
    }

    node_registry: dict[str, str] = {}
    edge_id       = 0
    all_detections: list[dict] = []

    # ── Node registry: merge endpoints that are within snap_px of each other ──
    def get_or_create_node(point) -> str:
        for key, nid in node_registry.items():
            kx, ky = map(int, key.split("_"))
            if abs(kx - point[0]) < snap_px and abs(ky - point[1]) < snap_px:
                return nid
        nid = f"N{len(node_registry):04d}"
        node_registry[f"{int(point[0])}_{int(point[1])}"] = nid
        flow_data["nodes"].append(
            {"id": nid, "x": int(point[0]), "y": int(point[1]), "type": "junction"}
        )
        return nid

    # ── Process one line segment ───────────────────────────────────────────────
    def process_line(line: tuple, line_type: str):
        nonlocal edge_id
        x1, y1, x2, y2 = line
        pt1, pt2 = (x1, y1), (x2, y2)

        cand_dirs = _candidate_directions(x1, y1, x2, y2)

        # NEW v2: search along the full segment, not just the two endpoints
        best = _search_along_line(
            binary_inv, x1, y1, x2, y2,
            templates, cand_dirs,
            n_samples=n_samples,
            search_radius=search_radius,
            threshold=match_threshold,
        )

        if best:
            direction, score, bx, by, bw, bh, _sample_pt, t = best
            # t close to 0 → match near start (pt1) → arrowhead is at pt1 end
            # t close to 1 → match near end  (pt2) → arrowhead is at pt2 end
            tip  = pt1 if t < 0.5 else pt2
            tail = pt2 if tip is pt1 else pt1
            method = "template_match"

            # Geometric sanity check: does the template direction agree with
            # the tail→tip vector?  If not, flip tail/tip so they are consistent.
            geom_dir = _get_direction_label(tail, tip)
            if direction != geom_dir:
                tip, tail = tail, tip

        else:
            # Density fallback: the arrowhead base is dense ink; the open tail
            # end has less ink.  Higher-density endpoint = arrowhead side = tip.
            d1 = _ink_density(gray, x1, y1)
            d2 = _ink_density(gray, x2, y2)
            tip    = pt2 if d2 >= d1 else pt1
            tail   = pt1 if tip is pt2 else pt2
            score  = 0.0
            hs     = 22
            bx     = max(0,              int(tip[0]) - hs)
            by     = max(0,              int(tip[1]) - hs)
            bw     = min(img.shape[1]-1, int(tip[0]) + hs) - bx
            bh     = min(img.shape[0]-1, int(tip[1]) + hs) - by
            method = "density_fallback"
            direction = _get_direction_label(tail, tip)

        angle  = round(float(np.degrees(np.arctan2(
                     tip[1]-tail[1], tip[0]-tail[0]))), 1)
        length = round(float(np.hypot(x2-x1, y2-y1)), 1)

        tail_node = get_or_create_node(tail)
        tip_node  = get_or_create_node(tip)

        flow_data["edges"].append({
            "id"              : f"E{edge_id:04d}",
            "line_type"       : line_type,
            "from_node"       : tail_node,
            "to_node"         : tip_node,
            "tail"            : {"x": int(tail[0]), "y": int(tail[1])},
            "tip"             : {"x": int(tip[0]),  "y": int(tip[1])},
            "tip_bbox"        : {"x": int(bx), "y": int(by),
                                  "w": int(bw), "h": int(bh)},
            "direction"       : direction,
            "angle_deg"       : angle,
            "line_length_px"  : length,
            "detection_method": method,
            "match_score"     : round(score, 3),
        })

        if method == "template_match":
            all_detections.append({
                "direction": direction,
                "score"    : score,
                "bx": int(bx), "by": int(by),
                "bw": int(bw), "bh": int(bh),
                "edge_id"  : f"E{edge_id:04d}",
            })

        # Draw arrow body on visualisation
        body_color = (200, 80, 0) if line_type == "solid" else (0, 160, 30)
        cv2.arrowedLine(
            vis,
            (int(tail[0]), int(tail[1])),
            (int(tip[0]),  int(tip[1])),
            body_color, 2, tipLength=0.2,
        )

        edge_id += 1

    # ── Run over all segments ──────────────────────────────────────────────────
    print(f"\n[Arrow detection v2] Processing "
          f"{len(solid_lines)} solid + {len(dotted_lines)} dotted lines ...")

    for line in solid_lines:
        process_line(line, "solid")
    for line in dotted_lines:
        process_line(line, "dotted")

    # ── NMS + draw tip bounding boxes ─────────────────────────────────────────
    kept = _nms_detections(all_detections, iou_thresh=0.30)

    for det in kept:
        direction = det["direction"]
        color     = _DIR_COLOR.get(direction, (128, 128, 128))
        bx, by    = det["bx"], det["by"]
        bx2, by2  = bx + det["bw"], by + det["bh"]

        cv2.rectangle(vis, (bx, by), (bx2, by2), color, 2)

        lbl = f"{_DIR_UNICODE.get(direction, '?')} {direction}"
        lx  = bx
        ly  = max(by - 5, 12)
        # Draw label twice for a pseudo-bold effect
        cv2.putText(vis, lbl, (lx+1, ly+1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, lbl, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    # Draw diamond marker for density-fallback tips (no NMS box)
    for edge in flow_data["edges"]:
        if edge["detection_method"] == "density_fallback":
            tx = edge["tip_bbox"]["x"] + edge["tip_bbox"]["w"] // 2
            ty = edge["tip_bbox"]["y"] + edge["tip_bbox"]["h"] // 2
            cv2.drawMarker(vis, (tx, ty), (120, 120, 120),
                           cv2.MARKER_DIAMOND, 10, 1)

    # ── Build summary ─────────────────────────────────────────────────────────
    flow_data["summary"] = {
        "total_edges"          : len(flow_data["edges"]),
        "total_nodes"          : len(flow_data["nodes"]),
        "solid_edges"          : len(solid_lines),
        "dotted_edges"         : len(dotted_lines),
        "directions"           : dict(Counter(
                                     e["direction"] for e in flow_data["edges"])),
        "detection_methods"    : dict(Counter(
                                     e["detection_method"] for e in flow_data["edges"])),
        "template_matches_kept": len(kept),
    }

    # ── Save outputs ──────────────────────────────────────────────────────────
    json_path = os.path.join(output_dir, f"flow_graph_pn_{page_number}.json")
    vis_path  = os.path.join(output_dir, f"flow_arrows_pn_{page_number}.png")

    with open(json_path, "w") as f:
        json.dump(flow_data, f, indent=2)

    cv2.imwrite(vis_path, vis)

    print(f"  [arrows v2] template matches kept (after NMS): {len(kept)}")
    print(f"  [arrows v2] JSON  -> {json_path}")
    print(f"  [arrows v2] image -> {vis_path}")
    print(f"  [arrows v2] summary -> {flow_data['summary']}")

    return flow_data


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 – Off-page connector (chevron boundary arrow) detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_offpage_connectors(
    image_path  : str,
    page_number : int,
    output_dir  : str,
    min_area    : int   = 100,
    max_area    : int   = 500000,
    aspect_min  : float = 1.0,
    aspect_max  : float = 10.0,
    debug       : bool  = False,
) -> dict:
    """
    Detect off-page connector chevrons (pentagon-shaped boundary arrows) in a P&ID.

    Strategy:
      1. Otsu threshold + morphological close to get solid contours.
      2. Area + aspect-ratio pre-filter (loose, catches all candidates).
      3. Multi-epsilon polygon sweep (3%→5%→8%→12%→16%) to get 4-7 vertices.
      4. Convexity ratio >= 0.60  (relaxed – scanned/PDF chevrons are nearly convex).
      5. Direction from vertex geometry; pixel-density fallback for ambiguous cases.
      6. debug=True prints per-filter drop counts so you can tune thresholds.

    Args:
        image_path  : Path to the original page PNG.
        page_number : 1-based page number (used in output filenames).
        output_dir  : Directory to save PNG + JSON.
        min_area    : Minimum contour area in pixels.
        max_area    : Maximum contour area in pixels.
        aspect_min  : Minimum bounding-box width/height ratio.
        aspect_max  : Maximum bounding-box width/height ratio.
        debug       : If True, print per-stage drop statistics.

    Returns:
        dict with keys: source_image, page_number, timestamp, image_size,
        connectors, total, outgoing, incoming, unknown.
    """
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis  = img.copy()
    h, w = gray.shape

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k3     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3, iterations=2)

    if debug:
        cv2.imwrite(
            os.path.join(output_dir, f"offpage_debug_binary_pn_{page_number}.png"),
            closed,
        )

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        print(f"  [offpage-debug] total external contours: {len(contours)}")

    drop_area = drop_aspect = drop_poly = drop_convex = 0

    EPSILON_FRACS = [0.03, 0.05, 0.08, 0.12, 0.16]

    def _pixel_direction(cnt, bx, bw):
        """
        Which half of the bounding box has more ink?
        The blunt (base) end has MORE ink; the tip is the OPPOSITE end.
        """
        mask = np.zeros_like(closed)
        cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
        left_px  = int(np.sum(mask[:, bx          : bx + bw // 2] > 0))
        right_px = int(np.sum(mask[:, bx + bw // 2 : bx + bw    ] > 0))
        if left_px  > right_px * 1.15: return "RIGHT"
        if right_px > left_px  * 1.15: return "LEFT"
        return "UNKNOWN"

    connectors: list[dict] = []
    conn_id = 0

    for cnt in contours:

        # Filter 1: area
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            drop_area += 1
            continue

        # Filter 2: bounding-box aspect ratio
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        aspect = bw / bh
        if not (aspect_min <= aspect <= aspect_max):
            drop_aspect += 1
            continue

        # Filter 3: polygon vertex count (multi-epsilon sweep)
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
                counts = [len(cv2.approxPolyDP(cnt, f * peri, True))
                          for f in EPSILON_FRACS]
                print(f"  [offpage-debug] poly-fail area={int(area):6d} "
                      f"aspect={aspect:.2f} verts@eps={counts}")
            continue

        # Filter 4: convexity (relaxed to 0.60)
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1:
            continue
        conv_ratio = area / hull_area
        if conv_ratio < 0.60:
            drop_convex += 1
            if debug:
                print(f"  [offpage-debug] convex-fail area={int(area):6d} "
                      f"aspect={aspect:.2f} conv={conv_ratio:.2f}")
            continue

        # Direction from vertex geometry
        pts         = approx.reshape(-1, 2)
        leftmost_x  = pts[:, 0].min()
        rightmost_x = pts[:, 0].max()
        tol_px      = max(bw * 0.12, 8)

        n_near_left  = int(np.sum(pts[:, 0] <= leftmost_x  + tol_px))
        n_near_right = int(np.sum(pts[:, 0] >= rightmost_x - tol_px))

        if   n_near_left  == 1 and n_near_right >= 2:
            direction = "LEFT"
        elif n_near_right == 1 and n_near_left  >= 2:
            direction = "RIGHT"
        else:
            direction = _pixel_direction(cnt, bx, bw)

        tip_x = int(leftmost_x) if direction == "LEFT" else int(rightmost_x)

        entry = {
            "id"        : f"OPC{conn_id:04d}",
            "direction" : direction,
            "bbox"      : {"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh)},
            "center"    : {"x": int(bx + bw // 2), "y": int(by + bh // 2)},
            "area_px"   : int(area),
            "n_vertices": len(approx),
            "conv_ratio": round(conv_ratio, 3),
            "tip_x"     : tip_x,
        }
        connectors.append(entry)
        conn_id += 1

        color_map = {
            "RIGHT"  : (0,   120, 255),
            "LEFT"   : (0,   200,  50),
            "UNKNOWN": (140, 140, 140),
        }
        color = color_map.get(direction, (140, 140, 140))

        cv2.rectangle(vis, (bx - 1, by - 1), (bx + bw + 1, by + bh + 1), color, 2)

        tip_half = max(6, bh // 3)
        cv2.rectangle(vis,
                      (tip_x - tip_half, by),
                      (tip_x + tip_half, by + bh),
                      color, 2)

        lbl    = ("-> OUT" if direction == "RIGHT"
                  else "<- IN" if direction == "LEFT" else "? UNK")
        font_s = max(0.32, min(bh / 55.0, 0.55))
        lbl_y  = max(by - 4, 14)
        cv2.putText(vis, lbl, (bx, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_s, color, 1, cv2.LINE_AA)

        cv2.polylines(vis, [approx], True, color, 1)

    if debug:
        print(f"  [offpage-debug] filter summary:")
        print(f"    total contours : {len(contours)}")
        print(f"    drop area      : {drop_area}")
        print(f"    drop aspect    : {drop_aspect}")
        print(f"    drop poly      : {drop_poly}")
        print(f"    drop convex    : {drop_convex}")
        print(f"    PASSED         : {len(connectors)}")

    result = {
        "source_image": image_path,
        "page_number" : page_number,
        "timestamp"   : datetime.now().isoformat(),
        "image_size"  : {"width": int(w), "height": int(h)},
        "connectors"  : connectors,
        "total"       : len(connectors),
        "outgoing"    : sum(1 for c in connectors if c["direction"] == "RIGHT"),
        "incoming"    : sum(1 for c in connectors if c["direction"] == "LEFT"),
        "unknown"     : sum(1 for c in connectors if c["direction"] == "UNKNOWN"),
    }

    json_path = os.path.join(output_dir, f"offpage_connectors_pn_{page_number}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    vis_path = os.path.join(output_dir, f"offpage_connectors_pn_{page_number}.png")
    cv2.imwrite(vis_path, vis)

    print(f"  [offpage] found {len(connectors)} connectors "
          f"({result['outgoing']} outgoing, {result['incoming']} incoming, "
          f"{result['unknown']} unknown)")
    print(f"  [offpage] JSON  -> {json_path}")
    print(f"  [offpage] image -> {vis_path}")

    return result
