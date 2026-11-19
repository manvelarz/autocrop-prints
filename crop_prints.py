#!/usr/bin/env python3
"""Split a batch of flatbed scans into one straightened file per print.

    python crop_prints.py SCAN_DIR OUT_DIR
    # deps: pip install opencv-python-headless numpy
"""
import sys
from pathlib import Path
import cv2
import numpy as np


def find_prints(gray, min_area=0.008):
    """Return print quads. Edge-based, so it also works on white-on-white."""
    h, w = gray.shape[:2]
    s = 2000 / max(h, w)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    area = small.shape[0] * small.shape[1]
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    med = float(np.median(blur))

    maps = [cv2.Canny(blur, int(0.66 * med), int(min(255, 1.33 * med)), L2gradient=True),
            cv2.Canny(blur, int(0.33 * med), int(med), L2gradient=True)]
    ad = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, min(small.shape[:2]) // 20 * 2 + 1, 7)
    maps.append(cv2.Canny(ad, 50, 150, L2gradient=True))

    quads = []
    for edges in maps:
        for k in (7, 21):
            e = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
            for c in cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
                if not area * min_area < cv2.contourArea(c) < area * 0.4:
                    continue
                hull = cv2.convexHull(c)          # a shadow line splits a contour, not a hull
                (cx, cy), (rw, rh), ang = cv2.minAreaRect(hull)
                if min(rw, rh) < 8 or max(rw, rh) / min(rw, rh) > 2.5:
                    continue
                if cv2.contourArea(hull) / (rw * rh) < 0.85:
                    continue
                quads.append(cv2.boxPoints(((cx, cy), (rw, rh), ang)) / s)
    # same print found by several passes: keep the first, drop the overlap
    kept = []
    for q in sorted(quads, key=lambda q: -cv2.contourArea(q.astype(np.int32))):
        b = cv2.boundingRect(q.astype(np.int32))
        if all(_overlap(b, cv2.boundingRect(k.astype(np.int32))) < 0.3
               and _inside(b, cv2.boundingRect(k.astype(np.int32))) < 0.6
               for k in kept):
            kept.append(q)
    kept.sort(key=lambda q: (cv2.boundingRect(q.astype(np.int32))[1] // 100,
                             cv2.boundingRect(q.astype(np.int32))[0]))
    return kept


def _overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    return i / (a[2] * a[3] + b[2] * b[3] - i)


def _inside(a, b):
    """How much of the smaller box sits inside the bigger one (catches a slice
    of a print, e.g. a strip split off by a shadow line)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1) / min(a[2] * a[3], b[2] * b[3])


def order(p):
    p = p.reshape(4, 2).astype(np.float32)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]], np.float32)


def crop(img, quad, margin=0.015):
    tl, tr, br, bl = order(quad)
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    dx, dy = w * margin, h * margin
    dst = np.array([[dx, dy], [w - dx, dy], [w - dx, h - dy], [dx, h - dy]], np.float32)
    m = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), dst)
    return cv2.warpPerspective(img, m, (int(w), int(h)),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def main(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    made = 0
    for f in sorted(p for p in src.iterdir() if p.suffix.lower() in {".tif", ".tiff", ".jpg", ".png"}):
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)      # keeps 16-bit TIFFs
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        det = gray.astype(np.uint8) if gray.dtype == np.uint8 else \
            cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        quads = find_prints(det)
        print(f"{f.name}: {len(quads)} print(s)")
        for i, q in enumerate(quads, 1):
            out = dst / f"{f.stem}_{i:02d}.tif"
            cv2.imwrite(str(out), crop(img, q),
                        [cv2.IMWRITE_TIFF_COMPRESSION, 5])   # LZW, never re-encode a scan lossy
            made += 1
    print(f"wrote {made} file(s) to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
