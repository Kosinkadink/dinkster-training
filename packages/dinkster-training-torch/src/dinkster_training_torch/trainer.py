"""Cold-loaded SD1.5 LoRA runtime backed by dinkster_inference."""

from __future__ import annotations

import math
from typing import Protocol, cast

import torch
import torch.nn.functional as functional
from dinkster_inference.model_base import BaseModel
from dinkster_inference.sd import load_checkpoint_guess_config

from .config import TrainingConfig
from .dataset import EncodedDataset, comfy_encoder_factory
from .lora_program import LoRAProgram
from .randomness import NamedRandomness


class _Sampling(Protocol):
    sigmas: torch.Tensor


class SD15LoRATrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.dataset = EncodedDataset(
            config.dataset,
            checkpoint_digest=config.checkpoint_digest,
            encoder_factory=comfy_encoder_factory(config.checkpoint_path),
        )

        model_patcher, _, _, _ = load_checkpoint_guess_config(
            str(config.checkpoint_path),
            output_vae=False,
            output_clip=False,
            output_clipvision=False,
            output_model=True,
            model_options={"dtype": torch.float32},
        )
        if model_patcher is None:
            raise ValueError("checkpoint does not contain an SD1.5 diffusion model")
        self.lora_program = LoRAProgram(
            model_patcher,
            rank=config.rank,
            alpha=config.alpha,
            seed=config.seed,
            target_patterns=config.target_patterns,
            device=self.device,
        )
        self.model = cast("BaseModel", self.lora_program.inject(self.device))
        self.optimizer = torch.optim.AdamW(
            self.lora_program.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.randomness = NamedRandomness(config.seed, self.device)
        self.step_cursor = 0
        self.data_cursor = 0
        self.last_loss: float | None = None

    def restore(
        self,
        *,
        adapter: dict[str, torch.Tensor],
        optimizer: dict[str, object],
        rng: dict[str, torch.Tensor],
        step_cursor: int,
        data_cursor: int,
        loss: float | None,
    ) -> None:
        self.lora_program.load_state_dict(adapter)
        self.optimizer.load_state_dict(optimizer)
        self.randomness.load_state_dict(rng)
        self.step_cursor = step_cursor
        self.data_cursor = data_cursor
        self.last_loss = loss

    def train_step(self) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        sampling = cast("_Sampling", self.model.model_sampling)
        for _ in range(self.config.gradient_accumulation_steps):
            batch = self.dataset.batch(
                self.data_cursor,
                self.config.batch_size,
                generator=self.randomness.data,
                device=self.device,
            )
            latents = self.model.process_latent_in(batch.latents.float())
            timesteps = torch.randint(
                len(sampling.sigmas),
                (latents.shape[0],),
                device=self.device,
                generator=self.randomness.timestep_noise,
            )
            noise = torch.randn(
                latents.shape,
                device=self.device,
                dtype=torch.float32,
                generator=self.randomness.timestep_noise,
            )
            sigmas = sampling.sigmas[timesteps].to(self.device)
            alpha = (1.0 / (1.0 + sigmas.square())).reshape(
                (latents.shape[0],) + (1,) * (latents.ndim - 1)
            )
            noisy = alpha.sqrt() * latents + (1.0 - alpha).sqrt() * noise
            prediction = self.model.diffusion_model(
                noisy,
                timesteps.float(),
                context=batch.text_embeddings.float(),
                transformer_options={},
            )
            loss = functional.mse_loss(prediction.float(), noise)
            (loss / self.config.gradient_accumulation_steps).backward()
            losses.append(loss.detach())
            self.data_cursor += 1
        self.optimizer.step()
        mean_loss = torch.stack(losses).mean().item()
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"training loss is not finite: {mean_loss}")
        self.step_cursor += 1
        self.last_loss = mean_loss
        return mean_loss

    def close(self) -> None:
        import dinkster_inference.model_management as model_management

        self.lora_program.close()
        model_management.unload_all_models()
