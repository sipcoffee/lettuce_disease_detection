"""Merge the per-class annotated folders into a single YOLO dataset with a train/val split."""
import random
import shutil
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent  # .../lettuce yolo
LETTUCE_DIR = ROOT.parent                      # .../Lettuce

SOURCES = [
    {
        "label_dir": ROOT / "walay_sakit_AN" / "labels" / "Train",
        "image_dir": LETTUCE_DIR / "naay_sakit_converted",
        "class_name": "healthy",
    },
    {
        "label_dir": ROOT / "leaf_spot_AN" / "labels" / "Train",
        "image_dir": LETTUCE_DIR / "naaysaket_converted",
        "class_name": "not healthy",
    },
]

OUT = ROOT / "dataset"
VAL_RATIO = 0.2

for split in ("train", "val"):
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

summary = {"train": {}, "val": {}}
missing = []

for src in SOURCES:
    label_files = sorted(src["label_dir"].glob("*.txt"))
    random.shuffle(label_files)
    n_val = max(1, round(len(label_files) * VAL_RATIO))
    val_set = set(label_files[:n_val])

    for lf in label_files:
        split = "val" if lf in val_set else "train"
        img_path = src["image_dir"] / (lf.stem + ".jpg")
        if not img_path.exists():
            missing.append(str(img_path))
            continue
        shutil.copy2(img_path, OUT / "images" / split / img_path.name)
        shutil.copy2(lf, OUT / "labels" / split / lf.name)
        summary[split][src["class_name"]] = summary[split].get(src["class_name"], 0) + 1

print("Copied counts:")
for split, counts in summary.items():
    print(f"  {split}: {counts} (total {sum(counts.values())})")

if missing:
    print(f"\nWARNING: {len(missing)} images were missing and skipped:")
    for m in missing:
        print(" ", m)

data_yaml = OUT / "data.yaml"
data_yaml.write_text(
    "path: {}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n"
    "  0: healthy\n"
    "  1: not healthy\n".format(OUT.as_posix())
)
print(f"\nWrote {data_yaml}")
