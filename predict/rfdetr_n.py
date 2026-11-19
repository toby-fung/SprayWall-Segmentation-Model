import os

from libreyolo import LibreYOLO

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.2"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

model = LibreYOLO("runs/train/rfdetrx_seg/weights/best.pt")

# Run prediction on a validation image and save the visualized output masks
result = model("spraywall_dataset_cleaned/images/valid/IMG_2839.png", save=True, device="cpu")

print("Inference complete! Check runs/detect/predict/ for visual outputs.")