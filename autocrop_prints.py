#!/usr/bin/env python3
"""
autocrop_prints.py — split flatbed / phone scans of multiple prints into
individual, straightened photo files.

Built for the hard case: prints sitting on a WHITE backing, where
thresholding/`-trim` merges the prints into the background. Detection is
edge-based (Canny + contours), rotation comes from minAreaRect, so tilted
or 90-degrees-off prints are handled.

Usage
-----
    python autocrop_prints.py INPUT_DIR                 # write to INPUT_DIR/cropped
    python autocrop_prints.py INPUT_DIR -o OUT_DIR \\
        --margin 0.015 --min-area 0.008 --max-dim 2000

    # just look at what it would cut, no files written:
    python autocrop_prints.py INPUT_DIR --dry-run

Outputs per input file  scan01.tif -> scan01_01.tif, scan01_02.tif, ...
plus contact_sheet.jpg (all crops of all files, numbered) so a whole batch
can be eyeballed in one screen. Originals are only ever read.

What it does NOT do: decide which way up the photo is. The crop is straightened
and perspective-corrected, but a print placed sideways on the glass comes out
sideways, exactly as scanned — telling a rotated photo from an upright one needs
content recognition, not geometry.

Requires: opencv-python-headless, numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------- geometry ---
def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def warp_quad(img: np.ndarray, quad: np.ndarray, margin: float) -> np.ndarray:
    """Perspective-warp a 4-point quad out of img, trimming `margin` per side."""
    tl, tr, br, bl = order_corners(quad)
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if w < 8 or h < 8:
        return np.empty((0, 0), img.dtype)
    # pull the destination corners inwards so the backing edge is dropped
    dx, dy = w * margin, h * margin
    dst = np.array(
        [[dx, dy], [w - dx, dy], [w - dx, h - dy], [dx, h - dy]], dtype=np.float32
    )
    m = cv2.getPerspectiveTransform(
        np.array([tl, tr, br, bl], dtype=np.float32), dst
    )
    out = cv2.warpPerspective(
        img, m, (int(round(w)), int(round(h))),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    # a print is never portrait-landscape ambiguous; keep the long side as is
    return out


def iou(a: tuple[float, float, float, float], b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def containment(a, b) -> float:
    """Fraction of the smaller box that lies inside the other one."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / min(aw * ah, bw * bh)


def _backing_level(small: np.ndarray) -> float:
    """Median brightness of the scan's outer frame = the backing/scanner lid."""
    b = max(2, int(round(min(small.shape[:2]) * 0.02)))
    frame = np.concatenate([
        small[:b, :].ravel(), small[-b:, :].ravel(),
        small[:, :b].ravel(), small[:, -b:].ravel(),
    ])
    return float(np.median(frame))


def _is_blank(small: np.ndarray, quad: np.ndarray, backing: float,
              tol: float = 14.0, limit: float = 0.80) -> bool:
    """True when the quad's interior is (almost) all backing colour.

    Kills phantom 'prints' made of empty backing between two real prints: a
    real photo is not uniform, and even a snowy/overcast one shows some
    variation away from the backing level.
    """
    mask = np.zeros(small.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 255)
    inside = small[mask > 0]
    if inside.size == 0:
        return True
    blank = np.abs(inside.astype(np.float32) - backing) <= tol
    return float(blank.mean()) > limit


# --------------------------------------------------------------- detection ---
def _edge_maps(small: np.ndarray) -> list[np.ndarray]:
    """Several edge maps, because one detector never covers every print.

    A pale print on white backing (overexposed sky, bleached paper) drops out of
    a global-threshold Canny pass while its neighbour is found easily, so each
    pass is unioned later. The adaptive pass is the workhorse for white-on-white.
    """
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    med = float(np.median(blur))
    clahe = cv2.createCLAHE(3.0, (8, 8)).apply(small)
    cmed = float(np.median(clahe))
    soft = cv2.GaussianBlur(small, (0, 0), max(1.0, min(small.shape[:2]) * 0.0015))
    smed = float(np.median(soft))

    def canny(base: np.ndarray, m: float, lo: float, hi: float) -> np.ndarray:
        return cv2.Canny(base, int(max(0, lo * m)), int(min(255, hi * m)),
                         apertureSize=3, L2gradient=True)

    maps = [
        canny(blur, med, 0.66, 1.33),
        canny(blur, med, 0.33, 1.00),
        canny(clahe, cmed, 0.66, 1.33),
        canny(soft, smed, 0.40, 1.10),
    ]
    # adaptive threshold -> Canny: the pass that finds prints whose border has
    # almost no contrast against the backing
    block = int(min(small.shape[:2]) * 0.05) | 1
    ad = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, block, 7)
    maps.append(cv2.Canny(ad, 50, 150, L2gradient=True))
    return maps


def find_quads(gray: np.ndarray, min_area: float, max_dim: int):
    """Return print quads, in reading order, in *gray* pixel coordinates."""
    h, w = gray.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    # 16-bit scans (common for flatbed TIFFs) are detected on an 8-bit copy;
    # the crop itself is still taken from the original depth
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray

    img_area = float(small.shape[0] * small.shape[1])
    backing = _backing_level(small)
    # Two closing sizes: the small one keeps touching prints apart, the large one
    # seals a badly broken border. Candidates from the small kernel are kept
    # first, so a pair merged by the large one gets dropped later as a slice of
    # two already-kept prints.
    kernels = sorted({max(3, int(round(min(small.shape[:2]) * f)) | 1) for f in (0.004, 0.010)})

    cands = []
    for edges in _edge_maps(small):
        for k in kernels:
            e = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
            e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=1)
            # RETR_LIST, not RETR_EXTERNAL: a print touched by a neighbouring
            # one must not be swallowed by the outer contour
            contours, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < img_area * min_area or area > img_area * 0.40:
                    continue
                # hull, not raw contour: a shadow line crossing a print splits
                # the contour but never splits its hull
                hull = cv2.convexHull(c)
                (cx, cy), (rw, rh), ang = cv2.minAreaRect(hull)
                if rw < 8 or rh < 8:
                    continue
                ar = max(rw, rh) / max(1e-6, min(rw, rh))
                if ar > 2.5:  # photo, not a strip of backing / scanner edge
                    continue
                box_area = rw * rh
                if cv2.contourArea(hull) / box_area < 0.85:
                    continue  # not rectangular enough to be a print
                peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
                if len(approx) == 4:  # keep real perspective when corners exist
                    quad = approx.reshape(4, 2).astype(np.float32)
                else:
                    quad = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.float32)
                cands.append((quad / scale, area / img_area, quad, k))

    # small kernel first, then biggest area: individual prints beat merges
    cands.sort(key=lambda t: (t[3], -t[1]))
    kept = []
    for quad, score, quad_small, _k in cands:
        x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
        box = (x, y, bw, bh)
        # a duplicate, or a slice of a print already kept (a shadow line or a
        # frame edge splits one print into a big box plus a thin strip)
        if any(iou(box, kbox) > 0.30 or containment(box, kbox) > 0.60
               for kbox, _ in kept):
            continue
        if _is_blank(small, quad_small, backing):
            continue  # a strip of the backing itself, not a print
        kept.append((box, quad))
    # reading order: top-to-bottom, then left-to-right, in rows
    kept.sort(key=lambda t: (round(t[0][1] / max(1, t[0][3])), t[0][0]))
    return [q for _, q in kept]


# ------------------------------------------------------------------ driver ---
def process_file(path: Path, out_dir: Path, args) -> list[Path]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  !! cannot read {path.name}", file=sys.stderr)
        return []
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    quads = find_quads(gray, args.min_area, args.max_dim)
    print(f"  {path.name}: {len(quads)} print(s) detected")
    if not quads:
        return []

    warped = [w for w in (warp_quad(img, q, args.margin) for q in quads) if w.size]
    # a box merged out of two touching prints is ~2x the batch's typical print,
    # so the split pass only fires on those (a panorama keeps its shape)
    med_long = float(np.median([max(w.shape[:2]) for w in warped])) if warped else 0.0

    crops = []
    for i, out in enumerate(warped, 1):
        halves = split_crop(out, args.margin, med_long) if med_long else []
        parts = halves if halves else [out]
        if halves:
            print(f"    (box {i} held two prints side by side — split)")
        for j, crop in enumerate(parts):
            if args.dry_run:
                crops.append(crop)
                continue
            suffix = f"{i:02d}" if len(parts) == 1 else f"{i:02d}{chr(96 + j + 1)}"
            dst = out_dir / f"{path.stem}_{suffix}{args.ext}"
            params = [cv2.IMWRITE_TIFF_COMPRESSION, 5] if args.ext in (".tif", ".tiff") else []
            if not cv2.imwrite(str(dst), crop, params):
                print(f"  !! write failed: {dst}", file=sys.stderr)
                continue
            crops.append(crop)
            print(f"    -> {dst.name}  {crop.shape[1]}x{crop.shape[0]}")
    return crops


def split_crop(crop: np.ndarray, margin: float, med_long: float,
               oversized: float = 1.4, peak_ratio: float = 3.0) -> list[np.ndarray]:
    """Split a crop that actually holds two prints placed shoulder to shoulder.

    Only attempted on boxes clearly bigger than the batch's typical print, so a
    strong internal line (a wall edge, a horizon) inside a genuine photo is never
    cut. Returns [] when no convincing seam is found.
    """
    h, w = crop.shape[:2]
    if med_long <= 0 or max(h, w) < oversized * med_long:
        return []
    g = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32)
    vertical = w >= h  # a wide box is split along a vertical seam
    prof = np.abs(np.diff(g, axis=1)).mean(axis=0) if vertical else np.abs(np.diff(g, axis=0)).mean(axis=1)
    n = len(prof)
    if n < 40:
        return []
    lo, hi = int(n * 0.25), int(n * 0.75)
    med = float(np.median(prof))
    if med <= 1e-6:
        return []
    idx = lo + int(np.argmax(prof[lo:hi]))
    if prof[idx] < peak_ratio * med:
        return []
    # both halves must be print-shaped, not slivers
    if not (0.2 * n < idx < 0.8 * n):
        return []
    dx = int(round((w if vertical else h) * margin))
    if vertical:
        a, b = crop[:, :max(1, idx - dx)], crop[:, min(w - 1, idx + dx):]
    else:
        a, b = crop[:max(1, idx - dx), :], crop[min(h - 1, idx + dx):, :]
    if a.size == 0 or b.size == 0:
        return []
    return [a, b]


def contact_sheet(crops, path: Path, cols: int = 6, thumb: int = 260, pad: int = 8):
    if not crops:
        return
    cells = []
    for i, c in enumerate(crops, 1):
        g = c if c.ndim == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        g = cv2.convertScaleAbs(g, alpha=255.0 / max(1, int(g.max()) or 1))
        g = g if g.ndim == 3 else cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        f = thumb / max(g.shape[:2])
        t = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        cell = np.full((thumb, thumb, 3), 40, np.uint8)
        y = (thumb - t.shape[0]) // 2
        x = (thumb - t.shape[1]) // 2
        cell[y:y + t.shape[0], x:x + t.shape[1]] = t
        cv2.putText(cell, str(i), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cells.append(cell)
    rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
    width = max(len(r) for r in rows) * (thumb + pad) + pad
    sheet = np.full((len(rows) * (thumb + pad) + pad, width, 3), 20, np.uint8)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            y = pad + r * (thumb + pad)
            x = pad + c * (thumb + pad)
            sheet[y:y + thumb, x:x + thumb] = cell
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"  contact sheet -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect, straighten and crop individual prints from scans.")
    ap.add_argument("input", type=Path, help="directory of scans (or a single file)")
    ap.add_argument("-o", "--out", type=Path, help="output directory (default: <input>/cropped)")
    ap.add_argument("--margin", type=float, default=0.015, help="inward trim per side, fraction (default 0.015)")
    ap.add_argument("--min-area", type=float, default=0.008, help="min print area as fraction of the scan (default 0.008)")
    ap.add_argument("--max-dim", type=int, default=2000, help="detection resolution cap on the long side (default 2000)")
    ap.add_argument("--ext", default=".tif", choices=[".tif", ".tiff", ".png", ".jpg"], help="output format (default .tif)")
    ap.add_argument("--sheet", default="contact_sheet.jpg", help="contact sheet filename (empty string to skip)")
    ap.add_argument("--sheet-cols", type=int, default=6, help="thumbnails per row on the contact sheet (default 6)")
    ap.add_argument("--dry-run", action="store_true", help="detect only, write a contact sheet of the crops")
    args = ap.parse_args()

    files = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.iterdir() if p.suffix.lower() in EXTS and p.is_file()
    )
    if not files:
        print("no image files found", file=sys.stderr)
        return 1

    out_dir = args.out or (args.input.parent / "cropped" if args.input.is_file() else args.input / "cropped")
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} file(s) -> {out_dir}")

    total = 0
    for f in files:
        crops = process_file(f, out_dir, args)
        total += len(crops)
        if args.sheet and crops:
            # one sheet per file, prefixed so a big batch stays readable
            base = Path(args.sheet)
            name = base.name if len(files) == 1 else f"{f.stem}_{base.name}"
            if args.dry_run:
                name = f"{f.stem}_dryrun_{base.name}"
                dest = base.with_name(name)
            else:
                dest = out_dir / name
            contact_sheet(crops, dest, cols=args.sheet_cols)

    print(f"\ndone: {total} print(s) from {len(files)} scan(s)")
    if args.dry_run:
        print("dry run — no crops written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
