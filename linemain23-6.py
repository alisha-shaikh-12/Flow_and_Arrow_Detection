# linemain23-6.py
# P&ID Line Detection – Main Pipeline
# Updated 23-Jun: uses improved detect_arrow_directions (v2) from linefunc23-6.py
#
# Pipeline:
#   Step 1  Extract single page image from PDF
#   Step 2  Morphological line separation (solid vs dotted)
#   Step 3  Hough line segment detection
#   Step 4  Arrow / flow-direction detection  ← UPDATED v2
#   Step 5  Off-page connector (chevron) detection


import os
import configparser
import math
import dataclasses

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Hot-reload the functions module so edits take effect without restarting ───
import importlib
import linefunc23_6 as line_detection_functions          # <-- matches filename
importlib.reload(line_detection_functions)

from linefunc23_6 import (
    extract_single_page_image,
    extract_lines_without_ocr,
    detect_hough_lines,
    save_morphology_results,
    detect_arrow_directions,
    detect_offpage_connectors,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Load config
# ═══════════════════════════════════════════════════════════════════════════════

config = configparser.ConfigParser()
config.read("config.ini")

pdf_path                       = config.get("DEFAULT", "pdf_path")
output_image_path              = config.get("DEFAULT", "output_image_path")
DPI                            = config.getint("DEFAULT", "DPI")
page_number                    = config.getint("DEFAULT", "page_number")
line_detection_morphology_output_path = config.get(
    "DEFAULT", "line_detection_morphology_output_path"
)
hough_transform_output_path    = config.get("DEFAULT", "hough_transform_output_path")
json_output_path               = config.get("DEFAULT", "json_output_path")
arrow_output_path              = config.get("DEFAULT", "arrow_detection_output_path")
offpage_output_path            = config.get(
    "DEFAULT", "offpage_detection_output_path",
    fallback="outputs/offpage_connectors",
)
gemini_api_key                 = config.get("DEFAULT", "gemini_api_key", fallback="")

# ── Arrow detection tuning (override here or add to config.ini) ───────────────
ARROW_MATCH_THRESHOLD = float(config.get("DEFAULT", "arrow_match_threshold", fallback="0.48"))
ARROW_SEARCH_RADIUS   = int(config.get("DEFAULT", "arrow_search_radius",   fallback="50"))
ARROW_N_SAMPLES       = int(config.get("DEFAULT", "arrow_n_samples",       fallback="7"))
#
# Tuning guide:
#   Missing arrows?   → lower ARROW_MATCH_THRESHOLD to 0.44, raise ARROW_N_SAMPLES to 9
#   Too many false?   → raise ARROW_MATCH_THRESHOLD to 0.54
#   Wrong direction?  → check HoughConfig.min_line_length (too small = bad endpoints)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 – Extract single page image from PDF
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Step 1: Extract page image ───────────────────────────────────────────")
image_path = extract_single_page_image(
    pdf_path,
    output_image_path=output_image_path,
    page_number=page_number,
)
print(f"  [page ] saved → {image_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 – Morphological line separation
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Step 2: Morphological line separation ────────────────────────────────")

# Kernel sizes derived from DPI so they scale correctly at any resolution:
#   dotted_kernel: targets ~10 px features  (DPI / 30)
#   solid_kernel : targets ~60 px features  (DPI /  5)
dotted_kernel = DPI // 30
solid_kernel  = DPI // 5

lines_dotted, lines_only_solid, lines_diff = extract_lines_without_ocr(
    image_path,
    dotted_kernel=dotted_kernel,
    solid_kernel=solid_kernel,
)

saved = save_morphology_results(
    page_number  = page_number,
    lines_solid  = lines_only_solid,
    lines_dotted = lines_dotted,
    lines_diff   = lines_diff,
    output_dir   = line_detection_morphology_output_path,
)

# Free the dotted-all mask — we only need the diff (dotted-only) from here on
del lines_dotted


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 – Hough line segment detection
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Step 3: Hough line detection ─────────────────────────────────────────")


@dataclasses.dataclass
class HoughConfig:
    """
    Hough transform parameters.  All units are in pixels at the image DPI.

    rho            : Distance resolution of the accumulator (px).  0.5 gives
                     sub-pixel precision.
    theta          : Angle resolution (radians).  π/1080 ≈ 0.17°.
    threshold      : Minimum accumulator votes to accept a line.
    min_line_length: Shortest segment to keep (px).  Lines shorter than this
                     are dropped — tune UP if you detect noise, DOWN if you
                     miss short stubs.
    max_line_gap   : Maximum gap (px) in a line that will be bridged.  Tune
                     UP if your solid lines are fragmented.
    canny_low      : Lower hysteresis threshold for Canny edge detection.
    canny_high     : Upper hysteresis threshold for Canny edge detection.
    """
    rho            : float = 0.5
    theta          : float = math.pi / 1080
    threshold      : int   = 50
    min_line_length: int   = 150
    max_line_gap   : int   = 40
    canny_low      : int   = 50
    canny_high     : int   = 150


lines_solid_total, lines_dotted_total = detect_hough_lines(
    lines_diff                  = lines_diff,
    lines_only_solid            = lines_only_solid,
    page_number                 = page_number,
    hough_transform_output_path = hough_transform_output_path,
    DPI                         = DPI,
    image_path                  = image_path,
    config                      = HoughConfig(),
)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 – Arrow / flow-direction detection  (v2 – updated 23-Jun)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Step 4: Arrow direction detection (v2) ───────────────────────────────")
print(f"   match_threshold = {ARROW_MATCH_THRESHOLD}")
print(f"   search_radius   = {ARROW_SEARCH_RADIUS} px")
print(f"   n_samples       = {ARROW_N_SAMPLES} points along each segment")

flow_data = detect_arrow_directions(
    image_path      = image_path,
    lines_solid     = lines_solid_total,
    lines_dotted    = lines_dotted_total,
    page_number     = page_number,
    output_dir      = arrow_output_path,
    gemini_api_key  = gemini_api_key,          # ignored, kept for compat
    match_threshold = ARROW_MATCH_THRESHOLD,
    search_radius   = ARROW_SEARCH_RADIUS,
    n_samples       = ARROW_N_SAMPLES,
)

print(f"\n  Pipeline step 4 complete.")
print(f"   Arrows image → {arrow_output_path}/flow_arrows_pn_{page_number}.png")
print(f"   JSON graph   → {arrow_output_path}/flow_graph_pn_{page_number}.json")
print(f"   Summary      → {flow_data['summary']}")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5 – Off-page connector (chevron boundary arrow) detection
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Step 5: Off-page connector detection ─────────────────────────────────")

offpage_data = detect_offpage_connectors(
    image_path   = image_path,
    page_number  = page_number,
    output_dir   = offpage_output_path,
    min_area     = 800,
    max_area     = 80000,
    aspect_min   = 1.2,
    aspect_max   = 6.0,
    debug        = True,
)

print(f"\n  Off-page connectors → {offpage_output_path}/offpage_connectors_pn_{page_number}.png")
print(f"  Summary: {offpage_data['total']} total | "
      f"{offpage_data['outgoing']} outgoing (→) | "
      f"{offpage_data['incoming']} incoming (←)")


# ═══════════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════════

print("\n══ Pipeline complete ═════════════════════════════════════════════════════")
print(f"  Page        : {page_number}")
print(f"  Solid lines : {len(lines_solid_total) if lines_solid_total is not None else 0}")
print(f"  Dotted lines: {len(lines_dotted_total) if lines_dotted_total is not None else 0}")
print(f"  Graph nodes : {flow_data['summary']['total_nodes']}")
print(f"  Graph edges : {flow_data['summary']['total_edges']}")
print(f"  Directions  : {flow_data['summary']['directions']}")
print(f"  Off-page    : {offpage_data['total']} connectors")
