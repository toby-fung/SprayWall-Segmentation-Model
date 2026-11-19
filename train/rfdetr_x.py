import os

from libreyolo import LibreYOLO

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.2"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

model = LibreYOLO("runs/train/rfdetrx_seg2/weights/best.pt")
model.train(
    data="spraywall_dataset_tiled/dataset.yaml",
    task="segment",
    imgsz=624,
    epochs=12,
    batch=1,
    lr0=1e-4,
    lr_drop=8,
    hsv_prob=0.5,
    degrees=8,
    translate=0.1,
    workers=4,
    patience=5,
    output_dir="runs/train/rfdetrx_seg2",
    device="cpu",
    resume=True,
)
