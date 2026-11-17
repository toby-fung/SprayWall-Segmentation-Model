from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRl-seg.pt")
model.train(
    data="spraywall_dataset_cleaned/dataset.yaml",
    epochs=50,
    imgsz=512,
    batch=8,
    lr0=1e-4,
    device=0,
)
