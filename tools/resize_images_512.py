#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def resize_to_512(input_path: Path, output_path: Path, *, size: int = 512) -> None:
    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image)
        image = center_crop_square(image)
        image = image.resize((size, size), resample=Image.LANCZOS)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        ext = output_path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            image.save(output_path, quality=95, optimize=True)
        else:
            image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize images to 512x512 (center-crop square + resize).")
    parser.add_argument("--input", type=Path, required=True, help="Input folder containing images")
    parser.add_argument("--output", type=Path, required=True, help="Output folder")
    parser.add_argument("--size", type=int, default=512, help="Target size (default: 512)")
    parser.add_argument("--recursive", action="store_true", help="Process images recursively")
    return parser.parse_args()


def iter_images(folder: Path, *, recursive: bool) -> list[Path]:
    if recursive:
        candidates = folder.rglob("*")
    else:
        candidates = folder.glob("*")

    images: list[Path] = []
    for p in candidates:
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            images.append(p)

    images.sort()
    return images


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input
    output_dir: Path = args.output

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder not found or not a directory: {input_dir}")

    images = iter_images(input_dir, recursive=bool(args.recursive))
    if not images:
        print(f"No images found in {input_dir}")
        return 0

    processed = 0
    for src in images:
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        try:
            resize_to_512(src, dst, size=int(args.size))
            processed += 1
        except Exception as e:
            print(f"[WARN] Failed: {src} -> {e}")

    print(f"Done. Processed {processed}/{len(images)} images -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
