import os

from libreyolo import LibreYOLO

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.2"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "rfdetrx_seg2"

model = LibreYOLO("runs/train/" + MODEL_NAME + "/weights/best.pt")

# Run prediction on a validation image and save the visualized output masks
result = model("spraywall_dataset/images/test/", save=True, device="cpu", output_path="runs/detect/" + MODEL_NAME + "/")

print("Inference complete! Check runs/detect/predict/ for visual outputs.")