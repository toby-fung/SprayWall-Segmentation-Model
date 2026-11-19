import os

from libreyolo import LibreYOLO

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.2"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

model = LibreYOLO("LibreRFDETRn-seg.pt")
model.train(
    data="spraywall_dataset_cleaned/dataset.yaml",
    task="segment",
    epochs=50,
    imgsz=312,
    batch=2,
    nbs=16,
    amp=True,
    amp_dtype="bfloat16",
    lr0=1e-4,
    workers=0,
    device="0",
    output_dir="runs/train/rfdetrn_seg",
)
