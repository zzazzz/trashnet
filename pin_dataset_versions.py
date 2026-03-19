"""
pin_dataset_versions.py
Dipanggil dari CI/CD pipeline sebelum kaggle kernels push.
Membaca versi terbaru dataset dari env var DATA_VERSION dan SCRIPT_VERSION,
lalu update kaggle_notebook/kernel-metadata.json agar Kaggle mount versi terbaru.
"""
import json
import os

metadata_path = "kaggle_notebook/kernel-metadata.json"

with open(metadata_path, "r") as f:
    meta = json.load(f)

dv = os.environ.get("DATA_VERSION", "").strip()
sv = os.environ.get("SCRIPT_VERSION", "").strip()

meta["dataset_sources"] = [
    "ziyadmuhammad/trashnet-training-script" + (":" + sv if sv else ""),
    "ziyadmuhammad/trashnet-data" + (":" + dv if dv else ""),
]

with open(metadata_path, "w") as f:
    json.dump(meta, f, indent=2)

print("kernel-metadata.json updated:")
for src in meta["dataset_sources"]:
    print(f"  - {src}")