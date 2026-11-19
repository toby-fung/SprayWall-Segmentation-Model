import os
from pathlib import Path

import numpy as np
from libreyolo import LibreYOLO

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.2"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

root, T, NC = Path("spraywall_dataset_tiled"), 624, 6
model = LibreYOLO("runs/train/rfdetrx_seg2/weights/best.pt")

def tolist(v):
    v = v.cpu() if hasattr(v, "cpu") else v
    return np.array(v).reshape(-1).tolist()

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)

cm = np.zeros((NC + 1, NC + 1), int)   # rows: true class (last = missed), cols: predicted (last = false positive)
for img in sorted((root / "valid" / "images").glob("*")):
    gts = []
    for line in (root / "valid" / "labels" / (img.stem + ".txt")).read_text().splitlines():
        p = line.split()
        if len(p) < 7: continue
        xy = np.array(p[1:], float).reshape(-1, 2) * T
        gts.append((int(p[0]), [xy[:,0].min(), xy[:,1].min(), xy[:,0].max(), xy[:,1].max()]))
    preds = []
    for b in model(str(img)).boxes:
        preds.append((int(tolist(b.cls)[0]), float(tolist(b.conf)[0]), tolist(b.xyxy)[:4]))
    preds.sort(key=lambda x: -x[1])
    used = set()
    for pc, conf, pb in preds:
        if conf < 0.3: continue
        best, bi = 0.5, None
        for i, (gc, gb) in enumerate(gts):
            if i not in used and iou(pb, gb) > best:
                best, bi = iou(pb, gb), i
        if bi is None: cm[NC, pc] += 1
        else: cm[gts[bi][0], pc] += 1; used.add(bi)
    for i, (gc, _) in enumerate(gts):
        if i not in used: cm[gc, NC] += 1

print("rows=true class (last row = false positives), cols=predicted (last col = missed)")
print(cm)