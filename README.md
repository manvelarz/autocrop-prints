# autocrop-prints

Split a folder of flatbed scans into **one straightened file per print** — built for the
awkward case: several photos lying on a **white backing**, where `magick -trim` and
thresholding merge everything into one blob.

![one scan becomes four files](demo_result.jpg)

Detection is edge-based (multi-pass Canny plus an adaptive-threshold pass), the rotation
comes from `minAreaRect` geometry, and every crop is perspective-warped straight out of
the original scan — no re-compression of the source, no lossy intermediate.

## Background

This started as a simple script.

I had boxes of old family photos and a flatbed scanner, and every scan came back with
several prints on it. Cropping them apart by hand, one by one, was not something I was
going to do a few thousand times, so I wrote a script: find each print on the scan,
straighten it, cut it out.

Then it grew. People who do not write scripts asked for it, so it became an app — no
terminal, no install, nothing uploaded: everything runs in the browser on your own
machine, the core cropping is free, and your photos never leave your device. Then it had
to work from a phone, and then the interface took on a life of its own — batch runs,
fixing corners by hand, editing, ZIP export. That app is
[SmartPhotoCropper](https://smartphotocropper.com), and the photo detection there is
done by a small neural network, which is what deals with piles of overlapping prints and
phone shots taken at an angle.

I opened this part up because it is genuinely useful to people who tinker. And to be
honest: even if you are technical, the app gives you more control — drag a corner that
the detector got wrong, run a whole batch, edit and export in one place — and it is
free and local either way. This repo is for when you want the machinery itself: drop it
in your own pipeline, change the detector, run it over a hundred thousand scans on a
server, or just take it apart to see how the geometry works.

What is here is the first step, kept as it was: no model, no browser, no account. OpenCV
only — edge detection, contours, `minAreaRect` for the rotation, a perspective warp per
print. Point it at a folder and it works.

## Quick start

```bash
pip install opencv-python-headless numpy

# whole folder -> <folder>/cropped/ , one TIFF per print + a contact sheet to check
python autocrop_prints.py /path/to/scans

# see what it would cut without writing anything
python autocrop_prints.py /path/to/scans --dry-run

# just want the one-file version? same detector, ~85 lines
python crop_prints.py /path/to/scans /path/to/out
```

| file | what it is |
|---|---|
| `autocrop_prints.py` | full CLI: batch folder, `--dry-run`, contact sheet per scan, 16-bit TIFF preserved, LZW output, split pass for two prints placed shoulder to shoulder |
| `crop_prints.py` | the same detector, compact — paste it anywhere, no flags |

## Flags (`autocrop_prints.py`)

| flag | default | what it does |
|---|---|---|
| `-o, --out` | `<input>/cropped` | output directory |
| `--margin` | `0.015` | inward trim per side, fraction — raises the leftover backing edge |
| `--min-area` | `0.008` | smallest print as a fraction of the scan |
| `--max-dim` | `2000` | detection resolution cap on the long side (detection only) |
| `--ext` | `.tif` | output format; `.tif` is LZW and keeps the original bit depth |
| `--sheet` / `--sheet-cols` | `contact_sheet.jpg` / `6` | QC sheet, one per input file |
| `--dry-run` | off | detect + contact sheet, no crops written |

Output names: `scan01.tif` → `scan01_01.tif`, `scan01_02.tif`, … A box holding two
touching prints is split and comes out as `scan01_04a.tif`, `scan01_04b.tif`.
**Originals are only ever read.**

## What it deliberately does not do

- **Does not decide which way up the photo is.** The crop is straightened and
  perspective-corrected, but a print placed sideways on the glass comes out sideways —
  telling a rotated photo from an upright one needs content recognition, not geometry.
- **Does not rescue a bleached print** with no visible border (an overexposed sky print
  on white paper): it usually finds a partial box. Fix that one by hand.
- **Does not guarantee the count.** Read the per-file count in the log and open the
  contact sheet. On 80 scans that glance takes two minutes and it is the whole QC step.

## Tested on

| scan | prints | found |
|---|---|---|
| 4 real flatbed scans, 9779×6864, 5–6 tilted prints each, white backing | 22 | 22 (6.4 s for the batch) |
| 4 prints, 90° off, white backing, 4819×6874 flatbed scan | 4 | 4 |
| synthetic, 4 tilted prints, white backing | 4 | 4 |
| synthetic, 6 prints incl. a touching pair and one bleached print | 6 | 5 (+1 partial) |
| single print, 16-bit TIFF | 1 | 1 (16-bit preserved) |

## License

MIT — see `LICENSE`. Use it, fork it, ship it inside something commercial; a link back
is nice but not required.
