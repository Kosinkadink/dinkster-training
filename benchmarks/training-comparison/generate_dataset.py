"""Generate the fixed synthetic image/caption comparison dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

CAPTIONS = (
    "a red circle above a blue striped field",
    "a green square beside two yellow circles",
    "three violet triangles on a pale background",
    "an orange arch over a cyan checkerboard",
    "a blue diamond centered in red rings",
    "four green bars crossing a yellow disk",
    "a violet spiral on an orange grid",
    "a cyan star above three red squares",
    "a yellow hexagon inside a blue frame",
    "two orange circles under green diagonal lines",
    "a red triangle centered on a cyan checkerboard",
    "five blue dots around a violet square",
    "a green ring over orange horizontal bands",
    "a yellow star inside a red diamond",
    "three cyan rectangles beside a violet disk",
    "an orange square above blue concentric circles",
)

PALETTE = (
    (224, 53, 68),
    (36, 110, 210),
    (49, 166, 104),
    (242, 190, 45),
    (147, 74, 190),
    (238, 126, 48),
    (47, 187, 201),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image(index: int, size: int) -> Image.Image:
    randomizer = random.Random(90210 + index)
    image = Image.new("RGB", (size, size), (238, 236, 226))
    draw = ImageDraw.Draw(image)
    spacing = 32 + index % 4 * 8
    background = PALETTE[(index + 2) % len(PALETTE)]
    for position in range(-size, size * 2, spacing):
        if index % 2:
            draw.line((position, 0, position + size, size), fill=background, width=8)
        else:
            draw.line((0, position, size, position), fill=background, width=8)
    foreground = PALETTE[index % len(PALETTE)]
    for shape_index in range(5):
        center_x = randomizer.randint(72, size - 72)
        center_y = randomizer.randint(72, size - 72)
        radius = randomizer.randint(28, 72)
        bounds = (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        )
        if (index + shape_index) % 3 == 0:
            draw.ellipse(bounds, fill=foreground, outline=(20, 20, 20), width=5)
        elif (index + shape_index) % 3 == 1:
            draw.rectangle(bounds, fill=foreground, outline=(20, 20, 20), width=5)
        else:
            points = tuple(
                (
                    center_x + math.cos(angle) * radius,
                    center_y + math.sin(angle) * radius,
                )
                for angle in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)
            )
            draw.polygon(points, fill=foreground, outline=(20, 20, 20), width=5)
    return image


def generate(output: Path, size: int) -> dict[str, object]:
    image_root = output / "1_compare"
    image_root.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    for index, caption in enumerate(CAPTIONS):
        image_path = image_root / f"{index:02d}.png"
        caption_path = image_root / f"{index:02d}.txt"
        _image(index, size).save(image_path, format="PNG", optimize=False)
        caption_path.write_text(caption + "\n", encoding="utf-8")
        files.extend(
            (
                {"path": image_path.relative_to(output).as_posix(), "sha256": _sha256(image_path)},
                {
                    "path": caption_path.relative_to(output).as_posix(),
                    "sha256": _sha256(caption_path),
                },
            )
        )
    manifest: dict[str, object] = {
        "generator": "dinkster-training-comparison-synthetic-v1",
        "image_count": len(CAPTIONS),
        "resolution": [size, size],
        "files": files,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    if args.size < 64:
        parser.error("--size must be at least 64")
    print(json.dumps(generate(args.output.resolve(), args.size), sort_keys=True))


if __name__ == "__main__":
    main()
