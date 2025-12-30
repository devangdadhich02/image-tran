import csv
import os
import shutil

ROOT = "/users/sedm6780/Code/VAE-DG-master/messidor_2"

CSV = os.path.join(ROOT, "train_labels.csv")
IMG_DIR = os.path.join(ROOT, "images")
OUT_DIR = "/users/sedm6780/Code/VAE-DG-master/DR/messidor2"

IMAGE_COL = "id_code"
LABEL_COL = "diagnosis"

def find_image(base):
    """Find image matching base name."""
    # Try the exact filename first (CSV may already include .tif)
    if os.path.exists(base):
        return base
    
    # Then try .tif and .TIF
    for ext in [".tif", ".TIF"]:
        path = base + ext
        if os.path.exists(path):
            return path
    return None

with open(CSV, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        base = row[IMAGE_COL].strip()

        # Remove extension if included
        base = os.path.splitext(base)[0]

        label = row[LABEL_COL].strip()

        full_base_path = os.path.join(IMG_DIR, base)

        src = find_image(full_base_path)

        if src is None:
            print(f"WARNING: missing image for {base}")
            continue

        dst_dir = os.path.join(OUT_DIR, label)
        os.makedirs(dst_dir, exist_ok=True)

        shutil.copy(src, os.path.join(dst_dir, os.path.basename(src)))

print("✔ DONE — Images sorted into DR/0–4")

