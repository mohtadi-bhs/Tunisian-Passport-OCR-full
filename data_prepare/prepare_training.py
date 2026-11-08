"""Prepare the YOLO11-OBB training dataset from the augmented passport data.

Steps:
  1. Build the train/val split with build_obb_dataset.py (split by original photo, so no leakage).
  2. Verify the result: image/label pairing, label format, coordinate range, class ids,
     leakage between train and val, and class coverage per split.
  3. Package the dataset as a zip for Google Drive (Unicode-safe, uses the zipfile module).

Usage:
    python prepare_training.py --src "C:\\...\\data_aug_new" --dst "C:\\...\\dataset_new" --seed 1 --zip

The script exits with code 1 if any check fails, so a bad dataset is never packaged.
"""
import argparse
import collections
import re
import subprocess
import sys
import zipfile
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}
AUG_RE = re.compile(r"_aug\d+$")
NUM_CLASSES = 15
RARE_CLASSES = {12}
BUILDER = Path(__file__).with_name("build_obb_dataset.py")


def base_of(stem):
    return AUG_RE.sub("", stem)


def build(src: Path, dst: Path, seed: int):
    if not BUILDER.exists():
        sys.exit(f"missing {BUILDER}: put build_obb_dataset.py in the same folder as this script")
    cmd = [sys.executable, str(BUILDER), "--src", str(src), "--dst", str(dst), "--seed", str(seed)]
    print("building split:", " ".join(f'"{c}"' if " " in c else c for c in cmd[1:]))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip())
        sys.exit("builder failed")


def verify(dst: Path):
    problems = []
    splits = {}
    for sp in ("train", "val"):
        img_dir, lab_dir = dst / "images" / sp, dst / "labels" / sp
        if not img_dir.is_dir() or not lab_dir.is_dir():
            problems.append(f"missing folder for {sp}")
            continue
        imgs = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT}
        labs = {p.stem for p in lab_dir.glob("*.txt")}
        if imgs != labs:
            problems.append(f"{sp}: {len(imgs - labs)} images without labels, {len(labs - imgs)} labels without images")

        cls_count = collections.Counter()
        for lab in lab_dir.glob("*.txt"):
            for ln_no, ln in enumerate(lab.read_text(encoding="utf-8").splitlines(), 1):
                if not ln.strip():
                    continue
                v = ln.split()
                if len(v) != 9:
                    problems.append(f"{lab.name}:{ln_no} has {len(v)} values, expected 9")
                    continue
                cls = int(v[0])
                if not 0 <= cls < NUM_CLASSES:
                    problems.append(f"{lab.name}:{ln_no} class id {cls} out of range")
                coords = [float(x) for x in v[1:]]
                if any(c < 0 or c > 1 for c in coords):
                    problems.append(f"{lab.name}:{ln_no} coordinate outside [0,1]")
                cls_count[cls] += 1
        splits[sp] = (imgs, cls_count)
        print(f"{sp}: {len(imgs)} images, {sum(cls_count.values())} objects, {len(cls_count)} classes")

    if "train" in splits and "val" in splits:
        tr_base = {base_of(s) for s in splits["train"][0]}
        va_base = {base_of(s) for s in splits["val"][0]}
        shared = tr_base & va_base
        if shared:
            problems.append(f"leakage: {len(shared)} original photos appear in both train and val")
        print(f"original photos: train {len(tr_base)}, val {len(va_base)}, shared {len(shared)}")

        for sp in ("train", "val"):
            missing = sorted(set(range(NUM_CLASSES)) - set(splits[sp][1]))
            if missing:
                level = "WARNING" if sp == "val" else "ERROR"
                print(f"{level}: classes absent from {sp}: {missing}")
                if sp == "train":
                    problems.append(f"classes absent from train: {missing}")
        rare_val = {c: splits["val"][1].get(c, 0) for c in RARE_CLASSES}
        if any(v == 0 for v in rare_val.values()):
            print("note: rare class(es) have no validation examples, so their scores will not be meaningful")
    return problems


def make_zip(dst: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(dst.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(dst.parent))
    size_mb = zip_path.stat().st_size / 1e6
    print(f"zip written: {zip_path} ({size_mb:.0f} MB)")
    if size_mb > 300:
        print("note: large zip; check free space on Google Drive before uploading")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="augmented data folder (images/ and labels/)")
    ap.add_argument("--dst", required=True, type=Path, help="output dataset folder")
    ap.add_argument("--seed", type=int, default=1, help="split seed (1 keeps class 12 in train)")
    ap.add_argument("--zip", action="store_true", help="also create <dst>.zip for Google Drive")
    args = ap.parse_args()

    if not (args.src / "images").is_dir() or not (args.src / "labels").is_dir():
        sys.exit(f"{args.src} must contain images/ and labels/ (run augment_obb.py first)")

    build(args.src, args.dst, args.seed)
    problems = verify(args.dst)
    if problems:
        print("\nCHECKS FAILED:")
        for p in problems[:30]:
            print("  -", p)
        if len(problems) > 30:
            print(f"  ... and {len(problems) - 30} more")
        sys.exit(1)
    print("\nall checks passed")

    if args.zip:
        make_zip(args.dst, args.dst.with_suffix(".zip"))
    print("data.yaml:", (args.dst / "data.yaml").resolve())


if __name__ == "__main__":
    main()