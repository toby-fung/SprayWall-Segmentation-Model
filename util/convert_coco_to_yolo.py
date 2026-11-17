#!/usr/bin/env python

from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="spraywall_dataset_cleaned/annotations/",
    save_dir="spraywall_dataset_cleaned/converted/",
    cls91to80=False, 
)

import json
import shutil
from pathlib import Path

from ultralytics.utils import YAML

converted_dir = Path("spraywall_dataset_cleaned/converted/labels")
dataset_dir = Path("spraywall_dataset_cleaned")

for src in converted_dir.iterdir():
    if not src.is_dir():
        continue
    dst = dataset_dir / "labels" / src.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.txt"):
        shutil.move(str(f), str(dst / f.name))

shutil.rmtree("spraywall_dataset_cleaned/converted", ignore_errors=True)

with open("spraywall_dataset_cleaned/annotations/instances_Train.json") as f:
    coco = json.load(f)

categories = sorted(coco["categories"], key=lambda x: x["id"])
names = {cat["id"] - 1: cat["name"] for cat in categories}

dataset = {
    "path": str(Path("spraywall_dataset_cleaned").resolve()),
    "train": "images/Train",
    "val": "images/Val",
    "names": names,
}

YAML.save("spraywall_dataset_cleaned/dataset.yaml", dataset)