"""Augment labelled passport images for YOLO11-OBB.

Input (flat folder from Label Studio, as in data.zip):
    <src>/images/*.jpg|png      <src>/labels/*.txt      (same stem)
    labels: YOLO-OBB lines "class x1 y1 x2 y2 x3 y3 x4 y4" (normalized), or 5-value "class cx cy w h"

Output (same flat layout, so build_obb_dataset.py can read it next):
    <dst>/images/<stem>.jpg          original (re-saved, so EXIF rotation is baked in)
    <dst>/images/<stem>_augK.jpg     augmented copies
    <dst>/labels/<stem>.txt, <stem>_augK.txt

Geometric transforms (rotation, scale, shift, perspective) move the field corners with the
same matrix as the pixels. A transform is rejected if any field leaves the frame, so no
field is ever cut off or silently dropped.

Photometric transforms (brightness, gamma, blur, noise, JPEG, glare, shadow, low resolution)
do not move anything, so labels are unchanged.

Horizontal flips are NOT used: a mirrored passport is not a valid document layout.

Paths with accents or other non-ASCII characters (e.g. "Projet par défaut") are supported:
images are read and written through NumPy, because cv2.imread/imwrite fail on such paths on Windows.

Usage:
    python augment_obb.py --src "C:\\path\\to\\data" --dst "C:\\path\\to\\data_aug_new" --copies 6 --rare-extra 6
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png"}
RARE_CLASSES = {12}  # place_of_birth_ar: very few examples; boosted with extra copies


def imread_any(path):
    """Read an image even when the path has accents or other non-ASCII characters (Windows).
    Returns a BGR array, or None if the file cannot be decoded. EXIF orientation is applied."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_any(path, img, params=None):
    """Write an image even when the path has non-ASCII characters. Raises if encoding fails."""
    ok, buf = cv2.imencode(Path(path).suffix, img, params or [])
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    buf.tofile(str(path))


def read_labels(path: Path):
    """Return list of (cls, pts) with pts normalized (4, 2). 5-value boxes become 4 corners."""
    objs = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        v = line.split()
        if not v:
            continue
        cls, vals = int(float(v[0])), [float(x) for x in v[1:]]
        if len(vals) == 8:
            pts = np.array(vals).reshape(4, 2)
        elif len(vals) == 4:
            cx, cy, w, h = vals
            pts = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                            [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
        else:
            raise ValueError(f"{path.name}: unsupported line with {len(v)} values")
        objs.append((cls, pts))
    return objs


def write_labels(path: Path, objs):
    """objs hold NORMALIZED corner points (0..1). Only clips and writes them."""
    lines = []
    for cls, pts in objs:
        p = np.clip(np.asarray(pts, dtype=np.float64), 0.0, 1.0)
        lines.append(f"{cls} " + " ".join(f"{x:.6f}" for x in p.flatten()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def random_geometric_matrix(w: int, h: int, rng: random.Random, max_rot: float, max_jitter: float):
    """3x3 matrix: small rotation + scale + shift, then mild perspective."""
    angle = rng.uniform(-max_rot, max_rot)
    scale = rng.uniform(0.9, 1.1)
    cx, cy = w / 2, h / 2
    A = cv2.getRotationMatrix2D((cx, cy), angle, scale)  # 2x3
    A = np.vstack([A, [0, 0, 1]])
    tx, ty = rng.uniform(-0.05, 0.05) * w, rng.uniform(-0.05, 0.05) * h
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[rng.uniform(-max_jitter, max_jitter) * w, rng.uniform(-max_jitter, max_jitter) * h]
                            for _ in range(4)])
    P = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    return P @ T @ A


def apply_geometric(img, objs, H):
    """objs are normalized; returns warped image and normalized objs, or (None, None) if a field leaves the frame."""
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    new_objs = []
    for cls, pts in objs:
        px = pts * [w, h]
        moved = cv2.perspectiveTransform(px.reshape(-1, 1, 2).astype(np.float32), H.astype(np.float32)).reshape(-1, 2)
        # every corner must stay inside the frame (with a 2 px safety margin)
        if (moved[:, 0] < 2).any() or (moved[:, 0] > w - 2).any() or \
           (moved[:, 1] < 2).any() or (moved[:, 1] > h - 2).any():
            return None, None
        new_objs.append((cls, moved / [w, h]))
    return warped, new_objs


def photometric(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.astype(np.float32)
    h, w = out.shape[:2]

    # brightness / contrast
    out = out * rng.uniform(0.75, 1.25) + rng.uniform(-35, 35)

    # white balance per channel
    out *= np.array([rng.uniform(0.92, 1.08) for _ in range(3)], dtype=np.float32)

    # glare: soft bright ellipse (phone flash / film reflection)
    if rng.random() < 0.35:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        gx, gy = rng.uniform(0, w), rng.uniform(0, h)
        sx, sy = rng.uniform(0.10, 0.30) * w, rng.uniform(0.05, 0.20) * h
        g = np.exp(-(((xx - gx) / sx) ** 2 + ((yy - gy) / sy) ** 2) / 2)
        out += (g * rng.uniform(40, 120))[..., None]

    # shadow: darken one side with a soft gradient
    if rng.random() < 0.2:
        t = rng.uniform(0.3, 0.7) * w
        xx = np.arange(w, dtype=np.float32)
        ramp = np.clip((xx - t) / (0.25 * w), 0, 1) if rng.random() < 0.5 else np.clip((t - xx) / (0.25 * w), 0, 1)
        out *= (1 - rng.uniform(0.2, 0.45) * ramp)[None, :, None]

    out = np.clip(out, 0, 255).astype(np.uint8)

    # gamma
    if rng.random() < 0.4:
        g = rng.uniform(0.7, 1.4)
        lut = np.array([((i / 255.0) ** (1 / g)) * 255 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(out, lut)

    # low resolution (phone photo far away / re-saved)
    if rng.random() < 0.2:
        s = rng.uniform(0.4, 0.7)
        out = cv2.resize(cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                         (w, h), interpolation=cv2.INTER_LINEAR)

    # blur
    if rng.random() < 0.4:
        sigma = rng.uniform(0.5, 1.5)
        out = cv2.GaussianBlur(out, (0, 0), sigma)

    # sensor noise
    if rng.random() < 0.3:
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 7), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # JPEG compression artefacts
    if rng.random() < 0.5:
        q = rng.randint(40, 90)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def augment_one(img, objs, rng, max_tries=12):
    """Try a few random geometric transforms until one keeps every field in frame."""
    h, w = img.shape[:2]
    for attempt in range(max_tries):
        # shrink the allowed rotation/jitter a little with each failed attempt
        k = 1.0 - 0.08 * attempt
        H = random_geometric_matrix(w, h, rng, max_rot=15 * k, max_jitter=0.04 * k)
        warped, new_objs = apply_geometric(img, objs, H)
        if warped is not None:
            return photometric(warped, rng), new_objs
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--copies", type=int, default=6, help="augmented copies per labelled image")
    ap.add_argument("--rare-extra", type=int, default=6, help="extra copies for images containing a rare class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    img_dir, lab_dir = args.src / "images", args.src / "labels"
    out_img, out_lab = args.dst / "images", args.dst / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    stats = {"originals": 0, "augmented": 0, "rejected": 0, "skipped_unlabeled": 0, "unreadable": 0}
    for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
        lab_path = lab_dir / f"{img_path.stem}.txt"
        if not lab_path.exists():
            stats["skipped_unlabeled"] += 1
            continue
        objs = read_labels(lab_path)
        if not objs:
            stats["skipped_unlabeled"] += 1
            continue

        img = imread_any(img_path)  # applies EXIF orientation
        if img is None:
            print(f"cannot read {img_path.name}, skipped")
            stats["unreadable"] += 1
            continue
        h, w = img.shape[:2]

        # original, re-saved so pixels and labels agree
        imwrite_any(out_img / f"{img_path.stem}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_labels(out_lab / f"{img_path.stem}.txt", objs)
        stats["originals"] += 1

        n_copies = args.copies
        if any(c in RARE_CLASSES for c, _ in objs):
            n_copies += args.rare_extra

        made = 0
        for k in range(1, n_copies + 1):
            aug_img, aug_objs = augment_one(img, objs, rng)
            if aug_img is None:
                stats["rejected"] += 1
                continue
            name = f"{img_path.stem}_aug{k}"
            imwrite_any(out_img / f"{name}.jpg", aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_labels(out_lab / f"{name}.txt", aug_objs)
            made += 1
        stats["augmented"] += made

    print("done:", stats)
    print("images in output:", len(list(out_img.iterdir())), "| labels:", len(list(out_lab.glob('*.txt'))))


if __name__ == "__main__":
    main()