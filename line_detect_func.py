## line_detection_functions.py

import os, math
import fitz
import cv2
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import Counter


# ── Page extraction ────────────────────────────────────────────────────────────

def extract_single_page_image(pdf_path, output_image_path, page_number, dpi=300):
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    os.makedirs(output_image_path, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        if not (1 <= page_number <= doc.page_count):
            raise ValueError(f"page {page_number} out of range")
        pix = doc.load_page(page_number-1).get_pixmap(
            matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
        out = os.path.join(output_image_path, f"page_{page_number}.png")
        pix.save(out)
    finally:
        doc.close()
    return os.path.abspath(out)


# ── Morphological line detection ───────────────────────────────────────────────

def extract_lines_without_ocr(image_path, dotted_kernel, solid_kernel):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    del img
    def _lines(src, k):
        return cv2.add(
            cv2.morphologyEx(src, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT,(k,1)), iterations=2),
            cv2.morphologyEx(src, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT,(1,k)), iterations=2))
    solid  = _lines(binary, solid_kernel)
    dotted = _lines(binary, dotted_kernel)
    del binary
    return dotted, solid, cv2.subtract(dotted, solid)


def save_morphology_results(page_number, lines_solid, lines_dotted, lines_diff, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved = {}
    for label, arr in [("solid",lines_solid),("dotted",lines_dotted),("diff",lines_diff)]:
        path = os.path.join(output_dir, f"lines_morphology_{label}_pn_{page_number}.png")
        cv2.imwrite(path, arr)
        saved[label] = path
        print(f"  [{label:>6}] saved → {path}")
    return saved


# ── Hough line detection ───────────────────────────────────────────────────────

def detect_hough_lines(lines_diff, lines_only_solid, image_path,
                       hough_transform_output_path, page_number, DPI, config):
    cfg = config
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    def _hough(mask):
        edges = cv2.Canny(mask, cfg.canny_low, cfg.canny_high, apertureSize=3)
        return cv2.HoughLinesP(edges, cfg.rho, cfg.theta, cfg.threshold,
                               minLineLength=cfg.min_line_length,
                               maxLineGap=cfg.max_line_gap)
    lines_dotted_total = _hough(lines_diff)
    lines_solid_total  = _hough(lines_only_solid)
    print(f"  [solid ] {len(lines_solid_total) if lines_solid_total is not None else 0} segment(s)")
    print(f"  [dotted] {len(lines_dotted_total) if lines_dotted_total is not None else 0} segment(s)")

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for segs,c in [(lines_dotted_total,(0,180,0)),(lines_solid_total,(180,0,0))]:
        if segs is not None:
            for s in segs:
                x1,y1,x2,y2=s[0]; cv2.line(vis,(x1,y1),(x2,y2),c,2)
    os.makedirs(hough_transform_output_path, exist_ok=True)
    cv2.imwrite(os.path.join(hough_transform_output_path,
                f"line_hough_transform_{page_number}.png"), vis)
    return lines_solid_total, lines_dotted_total


# ══════════════════════════════════════════════════════════════════════════════
# ARROW DETECTION  —  tile-based template matching + valve rejection
# ══════════════════════════════════════════════════════════════════════════════

def _make_arrow_templates(size):
    templates = {}
    for d in ("right","left","up","down"):
        img = np.zeros((size,size), dtype=np.uint8)
        h,w = size,size; cx,cy = w//2,h//2
        if   d=="right": pts=[[0,0],[0,h-1],[w-1,cy]]
        elif d=="left":  pts=[[w-1,0],[w-1,h-1],[0,cy]]
        elif d=="down":  pts=[[0,0],[w-1,0],[cx,h-1]]
        else:            pts=[[0,h-1],[w-1,h-1],[cx,0]]
        cv2.fillPoly(img,[np.array(pts)],255)
        img = cv2.GaussianBlur(img,(3,3),0.8)
        templates[d] = img
    return templates


def _direction_label(angle_deg):
    a = angle_deg % 360
    if a>180: a-=360
    if   -22.5<=a< 22.5: return "right"
    elif  22.5<=a< 67.5: return "up-right"
    elif  67.5<=a<112.5: return "up"
    elif 112.5<=a<157.5: return "up-left"
    elif a>=157.5 or a<-157.5: return "left"
    elif -157.5<=a<-112.5: return "down-left"
    elif -112.5<=a< -67.5: return "down"
    else: return "down-right"


def _angle_diff(a,b):
    d=abs(a-b)%360
    return min(d,360-d)


def _nms(pts, min_dist):
    pts=sorted(pts,key=lambda p:-p[0])
    kept=[]
    for p in pts:
        if not any(math.hypot(p[1]-k[1],p[2]-k[2])<min_dist for k in kept):
            kept.append(p)
    return kept


def _connectivity_check(cx,cy,tpl_dir,segments,snap_r,angle_tol=40.0):
    dir_angle={"right":0,"left":180,"up":90,"down":270}[tpl_dir]
    tail_angle=(dir_angle+180)%360
    tail_segs=[]; head_segs=[]
    for (x1,y1,x2,y2,lt) in segments:
        for (ex,ey,ox,oy) in [(x1,y1,x2,y2),(x2,y2,x1,y1)]:
            if (cx-ex)**2+(cy-ey)**2>snap_r**2: continue
            arrival=(math.degrees(math.atan2(-(ey-cy),ex-cx))+180)%360
            if _angle_diff(arrival,tail_angle)<=angle_tol:
                tail_segs.append((ex,ey,ox,oy,lt))
            elif _angle_diff(arrival,dir_angle)<=angle_tol:
                head_segs.append((ex,ey,ox,oy,lt))
    is_arrow=(len(tail_segs)>=1 and len(head_segs)==0)
    best=tail_segs[0] if tail_segs else (head_segs[0] if head_segs else None)
    return is_arrow,len(tail_segs),len(head_segs),best


def _safe(v):
    if isinstance(v,np.integer):  return int(v)
    if isinstance(v,np.floating): return float(v)
    if isinstance(v,list):        return [_safe(x) for x in v]
    if isinstance(v,dict):        return {k:_safe(u) for k,u in v.items()}
    return v


def _match_on_tile(binary_tile, templates, threshold, arrow_size, offset_x, offset_y):
    """Run template matching on a single tile, return candidates in global coords."""
    candidates = []
    for direction, tpl in templates.items():
        if tpl.shape[0]>binary_tile.shape[0] or tpl.shape[1]>binary_tile.shape[1]:
            continue
        result = cv2.matchTemplate(binary_tile, tpl, cv2.TM_CCOEFF_NORMED)
        h,w    = tpl.shape
        thr    = threshold
        locs   = np.where(result>=thr)
        if len(locs[0])>2000:          # auto-raise if too many on this tile
            thr = float(np.percentile(result[result>=thr], 60))
            locs = np.where(result>=thr)
        for y,x in zip(locs[0],locs[1]):
            score = float(result[y,x])
            cx    = x + w//2 + offset_x
            cy    = y + h//2 + offset_y
            candidates.append((score, cx, cy, direction))
    return candidates


def detect_arrows(
    image_path,
    lines_only_solid,
    lines_diff,
    page_number,
    output_dir,
    DPI,
    lines_solid_hough  = None,
    lines_dotted_hough = None,
    arrow_size         = 0,
    match_threshold    = 0.70,
    tile_size          = 800,   # px per tile (overlap = arrow_size*4)
    snap_radius        = 0,
    angle_tolerance    = 40.0,
) -> list[dict]:
    """
    Tile-based arrow detection with valve rejection.

    TILING: the full image is split into overlapping tiles so that:
      • Template matching runs on smaller patches → faster + more accurate
      • arrowheads at tile boundaries are not missed (due to overlap)

    VALVE REJECTION: matches where a pipe exists on BOTH sides are discarded.

    TUNING
    ───────
    arrow_size      : arrowhead template width (px). 0=auto (DPI//18).
    match_threshold : template confidence 0-1. Start 0.70.
    tile_size       : crop size per tile. 800px works well at 300 DPI.
    angle_tolerance : valve rejection strictness (°). 40 is standard.
    """
    scale = DPI/300
    if arrow_size  ==0: arrow_size   = max(8, int(17*scale))
    if snap_radius ==0: snap_radius  = max(20, arrow_size*3)
    overlap = arrow_size * 4

    print(f"  [arrows] DPI={DPI}  size={arrow_size}px  "
          f"threshold={match_threshold}  tile={tile_size}px  snap={snap_radius}px")

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    H,W = gray.shape
    _,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    os.makedirs(output_dir, exist_ok=True)
    templates = _make_arrow_templates(arrow_size)

    # Save template debug — scale up so it's visible
    tpl_vis=[]
    for d in ("right","left","up","down"):
        t=cv2.cvtColor(templates[d],cv2.COLOR_GRAY2BGR)
        cv2.putText(t,d[0].upper(),(1,arrow_size-2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,200,0),1)
        tpl_vis.append(t)
    sf = max(1, 80//arrow_size)
    cv2.imwrite(os.path.join(output_dir,"dbg_templates.png"),
                cv2.resize(np.hstack(tpl_vis),None,fx=sf,fy=sf,
                           interpolation=cv2.INTER_NEAREST))

    # Collect Hough segments
    all_segs=[]
    for segs,lt in [(lines_solid_hough,"solid"),(lines_dotted_hough,"dotted")]:
        if segs is None: continue
        for s in segs:
            x1,y1,x2,y2=int(s[0][0]),int(s[0][1]),int(s[0][2]),int(s[0][3])
            all_segs.append((x1,y1,x2,y2,lt))

    # ── Tile-based template matching ──────────────────────────────────────────
    all_candidates = []
    nms_dist = max(10, int(arrow_size*1.2))

    xs = list(range(0, W, tile_size-overlap))
    ys = list(range(0, H, tile_size-overlap))
    total_tiles = len(xs)*len(ys)
    print(f"  [arrows] processing {total_tiles} tiles ({len(xs)}×{len(ys)})...")

    for ti,ty0 in enumerate(ys):
        for tx0 in xs:
            tx1 = min(W, tx0+tile_size)
            ty1 = min(H, ty0+tile_size)
            tile = binary[ty0:ty1, tx0:tx1]
            candidates = _match_on_tile(tile, templates, match_threshold,
                                        arrow_size, tx0, ty0)
            all_candidates.extend(candidates)

    print(f"  [arrows] raw matches (all tiles): {len(all_candidates)}")

    # Global NMS across all tiles
    kept = _nms(all_candidates, nms_dist)
    print(f"  [arrows] after NMS: {len(kept)}")

    # ── Valve / arrow connectivity filter ────────────────────────────────────
    arrows=[]; n_valve=0; n_no_seg=0

    for score,cx,cy,tpl_dir in kept:
        is_arrow,tail_n,head_n,best_seg = _connectivity_check(
            cx,cy,tpl_dir,all_segs,snap_radius,angle_tolerance)
        if best_seg is None:
            n_no_seg+=1; continue
        if not is_arrow:
            n_valve+=1; continue
        ex,ey,ox,oy,lt=best_seg
        dx=ex-ox; dy=ey-oy
        hangle=math.degrees(math.atan2(-dy,dx))
        arrows.append({
            "tip"            :[cx,cy],
            "direction_angle":round(float(hangle%360),2),
            "direction_label":_direction_label(hangle),
            "match_score"    :round(score,3),
            "template_dir"   :tpl_dir,
            "line_type"      :lt,
            "tail_segments"  :tail_n,
            "head_segments"  :head_n,
        })

    dir_counts=Counter(a["direction_label"] for a in arrows)
    print(f"\n  [arrows] {len(arrows)} arrows  |  "
          f"{n_valve} valves rejected  |  {n_no_seg} isolated rejected")
    for d,n in sorted(dir_counts.items()):
        print(f"    {d:<14}: {n}")

    # ── Draw bounding boxes directly on a full-res BGR image ─────────────────
    # We do NOT use matplotlib for the main output — it downscales.
    # Instead we draw directly with cv2 and save a full-resolution PNG.
    BOX_COLOR={
        "right"     :(  0,200,  0),
        "left"      :(  0,  0,220),
        "up"        :(220, 80,  0),
        "down"      :(200,  0,200),
        "up-right"  :(  0,180,180),
        "up-left"   :(  0,160,220),
        "down-right":(180,  0,100),
        "down-left" :( 80, 80, 80),
    }
    SYM={"right":"->","left":"<-","up":"^","down":"v",
         "up-right":"^>","up-left":"^<","down-right":"v>","down-left":"v<"}

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    box_r    = max(14, DPI//20)       # half-side of bounding box
    lw       = max(2,  DPI//100)      # box line width
    font     = cv2.FONT_HERSHEY_SIMPLEX
    font_sc  = max(0.5,  DPI/500)     # larger font so labels are readable
    font_th  = max(1,    DPI//150)
    pad      = max(4,    DPI//80)

    for arr in arrows:
        c   = BOX_COLOR.get(arr["direction_label"],(80,80,80))
        cx  = int(arr["tip"][0]); cy=int(arr["tip"][1])
        sym = SYM.get(arr["direction_label"],"?")
        lbl = f"{sym} {arr['direction_label']} {arr['match_score']:.2f}"

        x1b=cx-box_r; x2b=cx+box_r
        y1b=cy-box_r; y2b=cy+box_r

        # Box
        cv2.rectangle(vis,(x1b,y1b),(x2b,y2b),c,lw)

        # Label bar above box
        (tw,th),_=cv2.getTextSize(lbl,font,font_sc,font_th)
        bar_x1=x1b; bar_x2=max(x2b, x1b+tw+pad*2)
        bar_y2=y1b; bar_y1=max(0, y1b-th-pad*2)
        bar_x2=min(W-1,bar_x2)

        cv2.rectangle(vis,(bar_x1,bar_y1),(bar_x2,bar_y2),c,-1)
        cv2.putText(vis,lbl,(bar_x1+pad,bar_y2-pad),
                    font,font_sc,(255,255,255),font_th,cv2.LINE_AA)

    # Save full-resolution PNG directly — NO matplotlib scaling
    out_png = os.path.join(output_dir,f"arrows_page_{page_number}.png")
    success = cv2.imwrite(out_png, vis)
    if not success:
        raise IOError(f"Failed to write {out_png}")
    print(f"\n  [arrows] full-res output → {out_png}")
    print(f"  [arrows] image size: {W}×{H} px  ({W/DPI*25.4:.0f}×{H/DPI*25.4:.0f} mm at {DPI} DPI)")

    # JSON
    out_json=os.path.join(output_dir,f"arrows_page_{page_number}.json")
    with open(out_json,"w") as f:
        json.dump([_safe(a) for a in arrows],f,indent=2)
    print(f"  [arrows] json → {out_json}")
    return arrows


def debug_arrow_detection(*args,**kwargs):
    print("Check dbg_templates.png — if template size looks wrong vs actual arrows,")
    print("set arrow_size explicitly: try DPI//15 (bigger) or DPI//22 (smaller).")