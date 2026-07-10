import os, configparser, math, dataclasses

import importlib, linefun2
importlib.reload(linefun2)
from linefun2 import (
    extract_single_page_image,
    extract_lines_without_ocr,
    save_morphology_results,
    detect_hough_lines,
    # detect_arrows,
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

# ── Step 1: Extract page image ──────────────────────────────────────────────
image_path = extract_single_page_image(pdf_path, out_img, page_number, dpi=DPI)

# ── Step 2: Morphological line separation ───────────────────────────────────
lines_dotted, lines_only_solid, lines_diff = extract_lines_without_ocr(
    image_path, dotted_kernel=DPI // 30, solid_kernel=DPI // 5)
save_morphology_results(page_number, lines_only_solid, lines_dotted, lines_diff, morph_out)
del lines_dotted

# ── Step 3: Hough segment detection ─────────────────────────────────────────
@dataclasses.dataclass
class HoughConfig:
    rho: float = 0.5
    theta: float = math.pi / 1080
    threshold: int = 50
    min_line_length: int = 150
    max_line_gap: int = 40
    canny_low: int = 50
    canny_high: int = 150

lines_solid_total, lines_dotted_total = detect_hough_lines(
    lines_diff, lines_only_solid, image_path,
    hough_out, page_number, DPI, HoughConfig())

# ── Step 4: Arrow detection ──────────────────────────────────────────────────
#
# TUNING GUIDE  (start here when results are wrong)
# ─────────────────────────────────────────────────
#  arrow_size      : 0 = auto (~20px at 300 DPI).
#                    → Check outputs/dbg_templates.png vs actual arrows.
#                    → Arrows look bigger than template? raise to DPI//12.
#                    → Template looks too big? lower to DPI//18.
#
#  match_threshold : default 0.75
#                    → Still too many false positives? raise to 0.78 or 0.80
#                    → Real arrows being missed?      lower to 0.72
#
#  angle_tolerance : default 30°  (stricter = fewer valve false positives)
#                    → Missing real arrows? raise to 35°
#                    → Valves still slipping through? lower to 25°
#
# arrows = detect_arrows(
#     image_path         = image_path,
#     lines_only_solid   = lines_only_solid,
#     lines_diff         = lines_diff,
#     page_number        = page_number,
#     output_dir         = arrow_out,
#     DPI                = DPI,
#     lines_solid_hough  = lines_solid_total,
#     lines_dotted_hough = lines_dotted_total,
#     # Hough lines PNG used as background → arrows + lines on ONE image
#     hough_vis_path     = os.path.join(hough_out, f"line_hough_transform_{page_number}.png"),
#     arrow_size         = 0,       # 0 = auto (~20px at 300 DPI); raise to DPI//12 if arrows are missed
#     match_threshold    = 0.70,    # lower → more arrows detected; raise to 0.75 to cut false positives
#     angle_tolerance    = 35.0,    # lower → stricter valve rejection; raise if real arrows are missed
# )

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n=== Page {page_number} summary ===")
print(f"  Solid  Hough segments : {len(lines_solid_total)  if lines_solid_total  is not None else 0}")
print(f"  Dotted Hough segments : {len(lines_dotted_total) if lines_dotted_total is not None else 0}")
# print(f"  Arrows detected       : {len(arrows)}")
print(f"\nOutput files:")
# print(f"  {arrow_out}/arrows_page_{page_number}.png    ← full-resolution visualisation")
# print(f"  {arrow_out}/arrows_page_{page_number}.json   ← structured arrow data")
# print(f"  {arrow_out}/dbg_templates.png                ← check template size vs real arrows")
# print(f"  {arrow_out}/dbg_binary_{page_number}.png     ← what the detector actually sees")