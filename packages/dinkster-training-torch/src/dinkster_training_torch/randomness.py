"""Named deterministic RNG streams stored in training checkpoints."""

from __future__ import annotations

import torch


class NamedRandomness:
    STREAMS = ("data", "timestep-noise")

    def __init__(self, seed: int, device: torch.device) -> None:
        timestep_device = device if device.type == "cuda" else torch.device("cpu")
        self._generators = {
            "data": torch.Generator(device="cpu").manual_seed(seed),
            "timestep-noise": torch.Generator(device=timestep_device).manual_seed(seed + 1),
        }

    @property
    def data(self) -> torch.Generator:
        return self._generators["data"]

    @property
    def timestep_noise(self) -> torch.Generator:
        return self._generators["timestep-noise"]

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: generator.get_state().cpu() for name, generator in self._generators.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if set(state) != set(self.STREAMS):
            raise ValueError(
                f"RNG checkpoint streams differ: expected {list(self.STREAMS)}, got {sorted(state)}"
            )
        for name, generator in self._generators.items():
            generator.set_state(state[name].cpu())
