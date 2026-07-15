"""Read-only facial image metadata and deterministic conversion helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps

from app.ml.common.hashing import sha256_file
from app.ml.preprocessing.face.constants import SUPPORTED_IMAGE_EXTENSIONS
from app.ml.preprocessing.face.schemas import FaceImageMetadata


def image_sha256(path: str | Path) -> str:
    return sha256_file(Path(path), allow_outside_project=True)


def extract_image_metadata(path: str | Path) -> FaceImageMetadata:
    image_path = Path(path)
    size = image_path.stat().st_size if image_path.exists() else 0
    digest = image_sha256(image_path) if image_path.exists() and image_path.is_file() else hashlib.sha256(b"").hexdigest()
    warnings: list[str] = []
    suffix = image_path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        warnings.append(f"unsupported image extension: {suffix or '<none>'}")
    if size == 0:
        warnings.append("zero-byte image file")
        return FaceImageMetadata(file_size_bytes=size, readable=False, image_hash=digest, validation_warnings=warnings)
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            return FaceImageMetadata(
                width=int(image.width),
                height=int(image.height),
                color_mode=str(image.mode),
                file_format=str(image.format or suffix.lstrip(".") or "unknown").lower(),
                file_size_bytes=size,
                readable=True,
                image_hash=digest,
                validation_warnings=warnings,
            )
    except Exception as exc:
        warnings.append(f"unreadable or corrupt image: {exc.__class__.__name__}")
        return FaceImageMetadata(file_size_bytes=size, readable=False, image_hash=digest, validation_warnings=warnings)


def convert_image_deterministic(
    source_path: str | Path,
    output_path: str | Path,
    *,
    target_width: int,
    target_height: int,
    color_mode: str,
    overwrite: bool = False,
    center_crop: bool = False,
) -> dict[str, object]:
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target dimensions must be positive")
    target = Path(output_path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing normalized image: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        if center_crop:
            image = ImageOps.fit(image, (target_width, target_height), method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))
            transform = "deterministic_center_crop_resize"
        else:
            image = ImageOps.pad(image, (target_width, target_height), method=Image.Resampling.BILINEAR, color=0, centering=(0.5, 0.5))
            transform = "deterministic_resize_pad"
        if color_mode.upper() == "RGB":
            image = image.convert("RGB")
        elif color_mode.upper() in {"L", "GRAYSCALE", "GRAY"}:
            image = image.convert("L")
        else:
            raise ValueError(f"Unsupported normalized image color mode: {color_mode}")
        image.save(target)
    return {
        "generated_image_relative_name": target.name,
        "target_width": target_width,
        "target_height": target_height,
        "color_mode": "L" if color_mode.upper() in {"L", "GRAYSCALE", "GRAY"} else "RGB",
        "transform": transform,
        "augmentation": "none",
    }

