#!/usr/bin/env python

import json
import shutil
from pathlib import Path


def remove_unlabelled_images(input_dir: str) -> None:
    base_path = Path(input_dir)

    json_dir = base_path / "annotations"
    if WANT_LABELLED:
        output_dir = Path(input_dir + "_labelled_only")
    else:
        output_dir = Path(input_dir + "_unlabelled_only")

    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON annotation files found in: {json_dir}")

    json_path = json_files[0]

    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    annotated_image_ids = {ann["image_id"] for ann in coco_data.get("annotations", [])}

    image_id_to_filename = {
        img["id"]: img["file_name"] for img in coco_data.get("images", [])
    }

    labelled_filenames = {
        image_id_to_filename[img_id]
        for img_id in annotated_image_ids
        if img_id in image_id_to_filename
    }

    if output_dir.exists():
        shutil.rmtree(output_dir)

    shutil.copytree(base_path, output_dir)

    for img_file in Path(output_dir / "images").rglob("*"):
        if img_file.is_file() and img_file.name not in labelled_filenames and (WANT_LABELLED and img_file.name not in labelled_filenames) or not WANT_LABELLED and img_file.name in labelled_filenames:
                img_file.unlink()

    print(f"Cleaned dataset at: {output_dir}")


if __name__ == "__main__":
    INPUT_DIR = "spraywall_dataset"
    WANT_LABELLED = False

    remove_unlabelled_images(INPUT_DIR)
