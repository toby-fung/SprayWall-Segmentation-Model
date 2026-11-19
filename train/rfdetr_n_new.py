import os

from libreyolo import LibreYOLO

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.2"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

model = LibreYOLO("LibreRFDETRn-seg.pt")
model.train(
    data="spraywall_dataset_cleaned/dataset.yaml",
    task="segment",
    epochs=50,                  # Run another 50 epochs at the higher resolution
    imgsz=516,                  # Step up from 312 to 416 (or 512 if it handles it)
    batch=1,                    # Drop physical batch to 1 to protect VRAM/APU memory
    nbs=16,                     # Keep nominal batch size 16 for stable gradients
    amp=True,
    amp_dtype="bfloat16",
    lr0=5e-5,                   # Lower learning rate slightly since we are fine-tuning further
    workers=0,
    device="0",               # Or "0" if you want to test if higher resolution holds up on GPU without hanging
    output_dir="runs/train/rfdetrn_seg",
    
    # --- Recommended Additions for Small Dataset Generalization ---
    degrees=5.0,          # Slight rotation variation
    translate=0.1,         # Image translation
    scale=0.3,             # Scaling variation
    weight_decay=0.0005,   # Regularization to fight overfitting
    warmup_epochs=3.0      # Stable learning rate warmup
)
