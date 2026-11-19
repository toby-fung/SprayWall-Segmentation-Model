import os

from libreyolo import LibreYOLO

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.2"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

model = LibreYOLO("LibreRFDETRn-seg.pt")

# Run prediction on a validation image and save the visualized output masks
result = model("spraywall_dataset_cleaned/images/Val/IMG_0861.jpg", save=True)

print("Inference complete! Check runs/detect/predict/ for visual outputs.")