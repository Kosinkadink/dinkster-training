"""Deterministic image-caption encoding through dinkster_inference."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
import torch.nn.functional as functional
from dinkster_api.v1 import digest_bytes
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file
from torchvision.transforms.functional import pil_to_tensor

from .config import DatasetConfig, canonical_json
from .durability import advisory_file_lock, atomic_replace, durable_mkdir

FORK_IDENTITY = "dinkster-inference@8eeb24bd5ef1e217929b699bd5f6b2d6b1b774f7"
PREPROCESS_IDENTITY = "exif-rgb-center-crop-bilinear-antialias/1"
_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    caption_path: Path
    relative_image: str
    digest: str


@dataclass(frozen=True)
class PreparedBatch:
    latents: torch.Tensor
    text_embeddings: torch.Tensor


class Encoder(Protocol):
    def __call__(self, images: torch.Tensor, captions: list[str]) -> PreparedBatch: ...


EncoderFactory = Callable[[], tuple[Encoder, Callable[[], None]]]


def inspect_dataset(settings: DatasetConfig) -> tuple[str, tuple[DatasetItem, ...]]:
    if not settings.root.is_dir():
        raise ValueError(f"dataset root does not exist: {settings.root}")
    items: list[DatasetItem] = []
    for image_path in sorted(
        path for path in settings.root.rglob("*") if path.suffix.lower() in _IMAGE_EXTENSIONS
    ):
        caption_path = image_path.with_suffix(settings.caption_extension)
        if not caption_path.is_file():
            raise ValueError(f"dataset image has no caption: {image_path}")
        relative = image_path.relative_to(settings.root).as_posix()
        digest = hashlib.sha256(
            image_path.read_bytes() + b"\0" + caption_path.read_bytes()
        ).hexdigest()
        items.append(DatasetItem(image_path, caption_path, relative, f"sha256:{digest}"))
    if not items:
        raise ValueError(f"dataset contains no supported images: {settings.root}")
    identity = digest_bytes(
        canonical_json(
            {
                "items": [{"path": item.relative_image, "digest": item.digest} for item in items],
                "resolution": list(settings.resolution),
                "preprocess": PREPROCESS_IDENTITY,
            }
        )
    )
    return identity, tuple(items)


def _pixels(item: DatasetItem, resolution: tuple[int, int]) -> torch.Tensor:
    target_height, target_width = resolution
    with Image.open(item.image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        if width * target_height > height * target_width:
            crop_width = height * target_width // target_height
            left = (width - crop_width) // 2
            image = image.crop((left, 0, left + crop_width, height))
        elif width * target_height < height * target_width:
            crop_height = width * target_height // target_width
            top = (height - crop_height) // 2
            image = image.crop((0, top, width, top + crop_height))
        pixels = pil_to_tensor(image).to(dtype=torch.float32).div_(255.0).unsqueeze(0)
    if tuple(pixels.shape[2:]) != resolution:
        pixels = functional.interpolate(
            pixels,
            size=resolution,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return pixels.squeeze(0).movedim(0, -1)


class EncodedDataset:
    def __init__(
        self,
        settings: DatasetConfig,
        *,
        checkpoint_digest: str,
        encoder_factory: EncoderFactory,
    ) -> None:
        dataset_digest, items = inspect_dataset(settings)
        cache_key = digest_bytes(
            canonical_json(
                {
                    "schemaVersion": 1,
                    "datasetDigest": dataset_digest,
                    "checkpointDigest": checkpoint_digest,
                    "encoder": FORK_IDENTITY,
                }
            )
        )
        cache_dir = settings.encoded_cache_root / "v1"
        durable_mkdir(cache_dir)
        cache_path = cache_dir / f"{cache_key[7:]}.safetensors"
        manifest_path = cache_dir / f"{cache_key[7:]}.json"
        with advisory_file_lock(cache_dir / f"{cache_key[7:]}.lock"):
            if not cache_path.exists() or not manifest_path.exists():
                images = torch.stack([_pixels(item, settings.resolution) for item in items])
                captions = [item.caption_path.read_text(encoding="utf-8").strip() for item in items]
                if any(not caption for caption in captions):
                    raise ValueError("dataset captions must not be empty")
                encoder, close = encoder_factory()
                try:
                    encoded = encoder(images, captions)
                finally:
                    close()
                if encoded.latents.shape[0] != len(items):
                    raise ValueError("VAE encoder returned the wrong batch size")
                if encoded.text_embeddings.shape[0] != len(items):
                    raise ValueError("text encoder returned the wrong batch size")
                tensors = {
                    "latents": encoded.latents.detach().float().cpu().contiguous(),
                    "text_embeddings": encoded.text_embeddings.detach().float().cpu().contiguous(),
                }
                temporary = cache_path.with_suffix(".tmp")
                save_file(tensors, temporary)
                data = temporary.read_bytes()
                temporary.unlink()
                atomic_replace(cache_path, data)
                atomic_replace(
                    manifest_path,
                    canonical_json(
                        {
                            "schemaVersion": 1,
                            "cacheKey": cache_key,
                            "datasetDigest": dataset_digest,
                            "itemCount": len(items),
                            "tensorDigest": hashlib.sha256(data).hexdigest(),
                        }
                    ),
                )
        manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="ascii")))
        if manifest.get("cacheKey") != cache_key or manifest.get("itemCount") != len(items):
            raise ValueError("encoded dataset cache manifest does not match its key")
        data = cache_path.read_bytes()
        if manifest.get("tensorDigest") != hashlib.sha256(data).hexdigest():
            raise ValueError("encoded dataset cache failed digest verification")
        tensors = load_file(cache_path, device="cpu")
        if set(tensors) != {"latents", "text_embeddings"}:
            raise ValueError("encoded dataset cache has unexpected tensors")
        self.dataset_digest = dataset_digest
        self._latents = tensors["latents"]
        self._text_embeddings = tensors["text_embeddings"]

    def batch(
        self,
        cursor: int,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> PreparedBatch:
        count = self._latents.shape[0]
        del cursor
        indices = torch.randint(count, (batch_size,), generator=generator).tolist()
        return PreparedBatch(
            latents=self._latents[indices].to(device=device),
            text_embeddings=self._text_embeddings[indices].to(device=device),
        )


def comfy_encoder_factory(checkpoint_path: Path) -> EncoderFactory:
    def load() -> tuple[Encoder, Callable[[], None]]:
        import dinkster_inference.model_management as model_management
        from dinkster_inference.sd import load_checkpoint_guess_config

        _, clip, vae, _ = load_checkpoint_guess_config(
            str(checkpoint_path),
            output_vae=True,
            output_clip=True,
            output_clipvision=False,
            output_model=False,
        )
        if clip is None or vae is None:
            raise ValueError("checkpoint must contain an SD1.5 VAE and text encoder")

        def encode(images: torch.Tensor, captions: list[str]) -> PreparedBatch:
            with torch.inference_mode():
                latents = vae.encode(images)
                embeddings = [cast("torch.Tensor", clip.encode(caption)) for caption in captions]
            return PreparedBatch(latents, torch.cat(embeddings, dim=0))

        return encode, model_management.unload_all_models

    return load
