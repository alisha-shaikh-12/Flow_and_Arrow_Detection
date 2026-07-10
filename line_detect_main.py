import os, configparser, math, dataclasses

import importlib, line_detect_func
importlib.reload(line_detect_func)
from line_detect_func import (
    extract_single_page_image,
    extract_lines_without_ocr,
    save_morphology_results,
    detect_hough_lines,
    detect_arrows,
)

cfg = configparser.ConfigParser()
cfg.read("config.ini")
pdf_path    = cfg.get("DEFAULT", "pdf_path")
out_img     = cfg.get("DEFAULT", "output_image_path")
page_number = cfg.getint("DEFAULT", "page_number")
morph_out   = cfg.get("DEFAULT", "line_detection_morphology_output_path")
hough_out   = cfg.get("DEFAULT", "hough_transform_output_path")
arrow_out   = cfg.get("DEFAULT", "arrow_detection_output_path")
DPI         = cfg.getint("DEFAULT", "DPI")

# Step 1: extract page
image_path = extract_single_page_image(pdf_path, out_img, page_number, dpi=DPI)

# Step 2: morphological line separation
lines_dotted, lines_only_solid, lines_diff = extract_lines_without_ocr(
    image_path, dotted_kernel=DPI//30, solid_kernel=DPI//5)
save_morphology_results(page_number, lines_only_solid, lines_dotted, lines_diff, morph_out)
del lines_dotted

# Step 3: Hough segments
@dataclasses.dataclass
class HoughConfig:
    rho:float=0.5; theta:float=math.pi/1080; threshold:int=50
    min_line_length:int=150; max_line_gap:int=40
    canny_low:int=50; canny_high:int=150

lines_solid_total, lines_dotted_total = detect_hough_lines(
    lines_diff, lines_only_solid, image_path, hough_out, page_number, DPI, HoughConfig())

# Step 4: Arrow detection
# ─────────────────────────────────────────────────────────────────
# Tile-based template matching + valve rejection.
# Output: arrows_page_N.png at FULL resolution (no matplotlib scaling).
#
# TUNE:
#   arrow_size      : arrowhead template size px. 0=auto (DPI//18 ≈ 17px).
#                     Run once, check dbg_templates.png vs actual arrows.
#                     Increase (e.g. DPI//14) if arrows are too small.
#   match_threshold : 0.70 is a good start.
#                     Raise to 0.75-0.80 if you see too many false detections.
#                     Lower to 0.65 if real arrows are being missed.
# ─────────────────────────────────────────────────────────────────
arrows = detect_arrows(
    image_path         = image_path,
    lines_only_solid   = lines_only_solid,
    lines_diff         = lines_diff,
    page_number        = page_number,
    output_dir         = arrow_out,
    DPI                = DPI,
    lines_solid_hough  = lines_solid_total,
    lines_dotted_hough = lines_dotted_total,
    arrow_size         = 0,     # 0 = auto
    match_threshold    = 0.70,
)

print(f"\n=== Page {page_number} ===")
print(f"  Arrows detected : {len(arrows)}")
print(f"\nOutput files:")
print(f"  {arrow_out}/arrows_page_{page_number}.png   ← full-resolution output")
print(f"  {arrow_out}/dbg_templates.png               ← check template size")
print(f"  {arrow_out}/arrows_page_{page_number}.json  ← structured data")