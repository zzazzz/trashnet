"""
pin_dataset_versions.py
Dipanggil dari CI/CD pipeline sebelum kaggle kernels push.
Selalu set dataset_sources tanpa nomor versi — Kaggle akan otomatis
pakai versi terbaru yang tersedia.
"""
import json
import os

metadata_path = "kaggle_notebook/kernel-metadata.json"

with open(metadata_path, "r") as f:
    meta = json.load(f)

# Selalu pakai tanpa pin versi — Kaggle mount versi terbaru secara otomatis
# Format dengan versi: "ziyadmuhammad/trashnet-data:3"
# Format tanpa versi : "ziyadmuhammad/trashnet-data"  <- selalu latest
meta["dataset_sources"] = [
    "ziyadmuhammad/trashnet-training-script",
    "ziyadmuhammad/trashnet-data",
]

with open(metadata_path, "w") as f:
    json.dump(meta, f, indent=2)

print("kernel-metadata.json updated:")
for src in meta["dataset_sources"]:
    print(f"  - {src}")