"""Build a leak-free YOLO11-OBB dataset from a flat folder of passport images and labels.

Input (flat layout, as in Data_aug or data_aug_new):
    <src>/images/*.jpg
    <src>/labels/*.txt          same stem as the image

Label formats handled (normalized to [0,1]):
    - 5 values : class cx cy w h            -> converted to 4 corners (axis-aligned OBB)
    - 9 values : class x1 y1 ... x4 y4      -> kept as is (already oriented)

Output (YOLO-OBB layout):
    <dst>/images/{train,val}/*.jpg
    <dst>/labels/{train,val}/*.txt         one line per object: class x1 y1 x2 y2 x3 y3 x4 y4
    <dst>/data.yaml

Split: all augmentations of one base photo (base_augN.jpg) go to the SAME split,
so an augmented copy in train never shares its original with val.

For a folder that is already split (src/train/... and src/val/...), pass --presplit
and the existing split is kept.

Paths with accents or other non-ASCII characters are supported: files are copied with
shutil and read with pathlib, which work on Windows with such paths.
"""
import argparse
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# Field names in the order of classes.txt from Label Studio (class id = line index)
NAMES = {i: n for i, n in enumerate([
    "address_ar", "date_of_birth", "date_of_expiry", "date_of_issue", "full_name_ar",
    "given_names", "identity_number", "issuing_place", "job_ar", "nationality",
    "passport_number", "place_of_birth", "place_of_birth_ar", "sex", "surname",
])}
AUG_RE = re.compile(r"_aug\d+$")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def base_of(stem: str) -> str:
    return AUG_RE.sub("", stem)


def to_obb_line(parts):
    cls = int(float(parts[0]))
    vals = [float(v) for v in parts[1:]]
    if len(vals) == 4:  # cx cy w h -> 4 corners, clockwise from top-left
        cx, cy, w, h = vals
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        pts = [x1, y1, x2, y1, x2, y2, x1, y2]
    elif len(vals) == 8:
        pts = vals
    else:
        raise ValueError(f"unsupported label with {len(vals)} coords: {parts}")
    pts = [min(max(p, 0.0), 1.0) for p in pts]  # keep inside the image
    return f"{cls} " + " ".join(f"{p:.6f}" for p in pts)


def read_label(path: Path):
    lines = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split()
        if parts:
            lines.append(to_obb_line(parts))
    return lines


def collect(src: Path):
    """Return ({stem: (image_path, [obb_lines], split_or_None)}, presplit_flag) for all labelled images."""
    items = {}
    presplit = (src / "train").is_dir() and (src / "val").is_dir()
    dirs = [(s, src / s) for s in ("train", "val")] if presplit else [(None, src)]
    for split, root in dirs:
        img_dir, lab_dir = root / "images", root / "labels"
        if not img_dir.is_dir() or not lab_dir.is_dir():
            raise SystemExit(f"missing folder: {img_dir} or {lab_dir}")
        images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT}
        for lab in sorted(lab_dir.glob("*.txt")):
            if lab.stem not in images:
                print(f"WARNING: no image for label {lab.name}, skipped")
                continue
            items[lab.stem] = (images[lab.stem], read_label(lab), split)
    return items, presplit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--presplit", action="store_true", help="keep an existing train/val split from src")
    args = ap.parse_args()

    items, presplit = collect(args.src)
    if not items:
        raise SystemExit(f"no labelled images found in {args.src}")

    if presplit and args.presplit:
        split_of = {s: items[s][2] for s in items}
        print("using existing train/val split (--presplit)")
    else:
        # group augmentations by base photo, then split the groups
        groups = defaultdict(list)
        for s in items:
            groups[base_of(s)].append(s)
        keys = sorted(groups)
        random.Random(args.seed).shuffle(keys)
        n_val = max(1, round(len(keys) * args.val_frac))
        val_keys = set(keys[:n_val])
        split_of = {s: ("val" if base_of(s) in val_keys else "train") for s in items}

    # sanity: no base photo in both splits
    by_base = defaultdict(set)
    for s in items:
        by_base[base_of(s)].add(split_of[s])
    shared = [b for b, sp in by_base.items() if len(sp) > 1]
    if shared:
        raise SystemExit(f"leakage: base photos in both splits: {shared[:5]}")

    if args.dst.exists():
        shutil.rmtree(args.dst)
    for sp in ("train", "val"):
        (args.dst / "images" / sp).mkdir(parents=True)
        (args.dst / "labels" / sp).mkdir(parents=True)

    cls_count = {"train": Counter(), "val": Counter()}
    for s, (img_path, lines, _) in items.items():
        sp = split_of[s]
        shutil.copy2(img_path, args.dst / "images" / sp / img_path.name)
        (args.dst / "labels" / sp / f"{s}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        for ln in lines:
            cls_count[sp][int(ln.split()[0])] += 1

    names_block = "\n".join(f"  {i}: {NAMES[i]}" for i in sorted(NAMES))
    (args.dst / "data.yaml").write_text(
        f"path: {args.dst.resolve().as_posix()}\n"
        f"train: images/train\nval: images/val\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )

    n_tr = sum(v == "train" for v in split_of.values())
    n_va = sum(v == "val" for v in split_of.values())
    print(f"base photos: {len(by_base)} | train images: {n_tr} | val images: {n_va}")
    for sp in ("train", "val"):
        print(f"{sp} objects per class:", dict(sorted(cls_count[sp].items())))
    missing_val = [c for c in NAMES if cls_count["val"][c] == 0]
    if missing_val:
        print("WARNING: classes absent from val:", missing_val, "-> try another --seed")


if __name__ == "__main__":
    main()