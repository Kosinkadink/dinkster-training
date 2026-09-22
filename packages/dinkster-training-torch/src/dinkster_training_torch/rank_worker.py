"""Spawned rank workers for durable multi-rank training sessions."""

from __future__ import annotations

import hashlib
import time
import traceback
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import cast

import torch

from .checkpoint import ContentAddressedCheckpointStore
from .config import (
    Flux2TrainingConfig,
    FluxTrainingConfig,
    Ideogram4TrainingConfig,
    MiniMaxH3TrainingConfig,
    MiniMaxMusic3TrainingConfig,
    QwenImageTrainingConfig,
    TrainingConfig,
    WanTrainingConfig,
)
from .dataset import TrainingFamily
from .distributed import TorchDistributedRankContext, bootstrap_rank_context
from .trainer import (
    DataSourceFactory,
    Flux2DataSourceFactory,
    Flux2LoRATrainer,
    Flux2ModelFactory,
    FluxDataSourceFactory,
    FluxLoRATrainer,
    FluxModelFactory,
    Ideogram4DataSourceFactory,
    Ideogram4LoRATrainer,
    Ideogram4ModelFactory,
    MiniMaxH3DataSourceFactory,
    MiniMaxH3LoRATrainer,
    MiniMaxH3ModelFactory,
    MiniMaxMusic3DataSourceFactory,
    MiniMaxMusic3LoRATrainer,
    MiniMaxMusic3ModelFactory,
    ModelFactory,
    QwenImageDataSourceFactory,
    QwenImageLoRATrainer,
    QwenImageModelFactory,
    RankContext,
    SD15LoRATrainer,
    SDXLLoRATrainer,
    WanDataSourceFactory,
    WanLoRATrainer,
    WanModelFactory,
)

_TrainingConfiguration = (
    TrainingConfig
    | FluxTrainingConfig
    | Flux2TrainingConfig
    | MiniMaxH3TrainingConfig
    | MiniMaxMusic3TrainingConfig
    | QwenImageTrainingConfig
    | Ideogram4TrainingConfig
    | WanTrainingConfig
)
_Trainer = (
    SD15LoRATrainer
    | SDXLLoRATrainer
    | FluxLoRATrainer
    | Flux2LoRATrainer
    | MiniMaxH3LoRATrainer
    | MiniMaxMusic3LoRATrainer
    | QwenImageLoRATrainer
    | Ideogram4LoRATrainer
    | WanLoRATrainer
)
_TrainerType = (
    type[SD15LoRATrainer]
    | type[SDXLLoRATrainer]
    | type[FluxLoRATrainer]
    | type[Flux2LoRATrainer]
    | type[MiniMaxH3LoRATrainer]
    | type[MiniMaxMusic3LoRATrainer]
    | type[QwenImageLoRATrainer]
    | type[Ideogram4LoRATrainer]
    | type[WanLoRATrainer]
)
_ModelFactory = (
    ModelFactory
    | FluxModelFactory
    | Flux2ModelFactory
    | MiniMaxH3ModelFactory
    | MiniMaxMusic3ModelFactory
    | QwenImageModelFactory
    | Ideogram4ModelFactory
    | WanModelFactory
)
_DataSourceFactory = (
    DataSourceFactory
    | FluxDataSourceFactory
    | Flux2DataSourceFactory
    | MiniMaxH3DataSourceFactory
    | MiniMaxMusic3DataSourceFactory
    | QwenImageDataSourceFactory
    | Ideogram4DataSourceFactory
    | WanDataSourceFactory
)


class RankWorkerError(RuntimeError):
    """A spawned training rank failed a command or exited unexpectedly."""


def config_for_rank(config: _TrainingConfiguration, rank: int) -> _TrainingConfiguration:
    """Map the shared CUDA selector to one rank-local CUDA device."""
    settings = config.distributed
    if settings is None or settings.world_size == 1 or config.device == "cpu":
        return config
    if config.device not in ("cuda", "cuda:0"):
        raise ValueError(
            f"multi-rank training supports device 'cpu', 'cuda', or 'cuda:0', not {config.device!r}"
        )
    return replace(config, device=f"cuda:{rank}")


def _set_rank_local_cuda_device(config: _TrainingConfiguration) -> None:
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)


def adapter_sync_digest(trainer: _Trainer) -> str:
    """Hash adapter parameter bytes in their stable attachment order."""
    digest = hashlib.sha256()
    parameters = trainer.attachment.parameters()
    if parameters:
        value = torch.cat(tuple(parameter.detach().reshape(-1) for parameter in parameters)).to(
            device="cpu"
        )
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _config_from_mapping(
    mapping: dict[str, object], expected_family: str
) -> _TrainingConfiguration:
    if expected_family == "minimax-h3":
        return MiniMaxH3TrainingConfig.from_mapping(mapping)
    if expected_family == "minimax-music3":
        return MiniMaxMusic3TrainingConfig.from_mapping(mapping)
    if expected_family == "wan":
        return WanTrainingConfig.from_mapping(mapping)
    if expected_family == "flux":
        return FluxTrainingConfig.from_mapping(mapping)
    if expected_family == "flux2":
        return Flux2TrainingConfig.from_mapping(mapping)
    if expected_family == "qwen-image":
        return QwenImageTrainingConfig.from_mapping(mapping)
    if expected_family == "ideogram4":
        return Ideogram4TrainingConfig.from_mapping(mapping)
    return TrainingConfig.from_mapping(
        mapping,
        expected_family=cast("TrainingFamily", expected_family),
    )


def build_trainer(
    config: _TrainingConfiguration,
    *,
    trainer_type: _TrainerType,
    model_factory: _ModelFactory,
    data_source_factory: _DataSourceFactory,
    rank_context: RankContext | None = None,
) -> _Trainer:
    local_config = config if rank_context is None else config_for_rank(config, rank_context.rank)
    if isinstance(local_config, FluxTrainingConfig):
        flux_trainer_type = cast("type[FluxLoRATrainer]", trainer_type)
        flux_model_factory = cast("FluxModelFactory", model_factory)
        flux_data_source_factory = cast("FluxDataSourceFactory", data_source_factory)
        if local_config.dataset is not None:
            data_source = flux_data_source_factory(local_config)
            model = flux_model_factory(local_config)
        else:
            model = flux_model_factory(local_config)
            data_source = flux_data_source_factory(local_config)
        if rank_context is None:
            return flux_trainer_type(local_config, model, data_source)
        return flux_trainer_type(
            local_config,
            model,
            data_source,
            rank_context=rank_context,
        )
    if isinstance(local_config, Flux2TrainingConfig):
        flux2_trainer_type = cast("type[Flux2LoRATrainer]", trainer_type)
        flux2_model_factory = cast("Flux2ModelFactory", model_factory)
        flux2_data_source_factory = cast("Flux2DataSourceFactory", data_source_factory)
        if local_config.dataset is not None:
            data_source = flux2_data_source_factory(local_config)
            model = flux2_model_factory(local_config)
        else:
            model = flux2_model_factory(local_config)
            data_source = flux2_data_source_factory(local_config)
        if rank_context is None:
            return flux2_trainer_type(local_config, model, data_source)
        return flux2_trainer_type(
            local_config,
            model,
            data_source,
            rank_context=rank_context,
        )
    if isinstance(local_config, QwenImageTrainingConfig):
        qwen_trainer_type = cast("type[QwenImageLoRATrainer]", trainer_type)
        qwen_model_factory = cast("QwenImageModelFactory", model_factory)
        qwen_data_source_factory = cast("QwenImageDataSourceFactory", data_source_factory)
        if local_config.dataset is not None:
            data_source = qwen_data_source_factory(local_config)
            model = qwen_model_factory(local_config)
        else:
            model = qwen_model_factory(local_config)
            data_source = qwen_data_source_factory(local_config)
        if rank_context is None:
            return qwen_trainer_type(local_config, model, data_source)
        return qwen_trainer_type(
            local_config,
            model,
            data_source,
            rank_context=rank_context,
        )
    if isinstance(local_config, Ideogram4TrainingConfig):
        ideogram_trainer_type = cast("type[Ideogram4LoRATrainer]", trainer_type)
        ideogram_model_factory = cast("Ideogram4ModelFactory", model_factory)
        ideogram_data_source_factory = cast("Ideogram4DataSourceFactory", data_source_factory)
        if local_config.dataset is not None:
            data_source = ideogram_data_source_factory(local_config)
            model = ideogram_model_factory(local_config)
        else:
            model = ideogram_model_factory(local_config)
            data_source = ideogram_data_source_factory(local_config)
        if rank_context is None:
            return ideogram_trainer_type(local_config, model, data_source)
        return ideogram_trainer_type(
            local_config,
            model,
            data_source,
            rank_context=rank_context,
        )
    if isinstance(local_config, WanTrainingConfig):
        wan_trainer_type = cast("type[WanLoRATrainer]", trainer_type)
        wan_model_factory = cast("WanModelFactory", model_factory)
        wan_data_source_factory = cast("WanDataSourceFactory", data_source_factory)
        if local_config.dataset is not None:
            data_source = wan_data_source_factory(local_config)
            model = wan_model_factory(local_config)
        else:
            model = wan_model_factory(local_config)
            data_source = wan_data_source_factory(local_config)
        if rank_context is None:
            return wan_trainer_type(local_config, model, data_source)
        return wan_trainer_type(
            local_config,
            model,
            data_source,
            rank_context=rank_context,
        )
    if isinstance(local_config, MiniMaxH3TrainingConfig):
        h3_trainer_type = cast("type[MiniMaxH3LoRATrainer]", trainer_type)
        h3_model_factory = cast("MiniMaxH3ModelFactory", model_factory)
        h3_data_source_factory = cast("MiniMaxH3DataSourceFactory", data_source_factory)
        if local_config.dataset is not None:
            data_source = h3_data_source_factory(local_config)
            model = h3_model_factory(local_config)
        else:
            model = h3_model_factory(local_config)
            data_source = h3_data_source_factory(local_config)
        if rank_context is None:
            return h3_trainer_type(local_config, model, data_source)
        return h3_trainer_type(
            local_config,
            model,
            data_source,
            rank_context=rank_context,
        )
    if isinstance(local_config, MiniMaxMusic3TrainingConfig):
        music3_trainer_type = cast("type[MiniMaxMusic3LoRATrainer]", trainer_type)
        music3_model_factory = cast("MiniMaxMusic3ModelFactory", model_factory)
        music3_data_source_factory = cast("MiniMaxMusic3DataSourceFactory", data_source_factory)
        data_source = music3_data_source_factory(local_config)
        model = music3_model_factory(local_config)
        if rank_context is None:
            return music3_trainer_type(local_config, model, data_source)
        return music3_trainer_type(
            local_config,
            model,
            data_source,
            rank_context=rank_context,
        )
    sd_trainer_type = cast("type[SD15LoRATrainer] | type[SDXLLoRATrainer]", trainer_type)
    sd_model_factory = cast("ModelFactory", model_factory)
    sd_data_source_factory = cast("DataSourceFactory", data_source_factory)
    model = sd_model_factory(local_config)
    data_source = sd_data_source_factory(local_config)
    if rank_context is None:
        return sd_trainer_type(local_config, model, data_source)
    return sd_trainer_type(
        local_config,
        model,
        data_source,
        rank_context=rank_context,
    )


def _load_worker_trainer(
    rank: int,
    payload: object,
    *,
    expected_family: str,
    trainer_type: _TrainerType,
    model_factory: _ModelFactory,
    data_source_factory: _DataSourceFactory,
) -> tuple[_Trainer, TorchDistributedRankContext]:
    if not isinstance(payload, tuple) or len(payload) != 3:
        raise TypeError("load command payload is malformed")
    mapping_value, checkpoint_digest, checkpoint_root = payload
    if not isinstance(mapping_value, dict) or not all(
        isinstance(key, str) for key in mapping_value
    ):
        raise TypeError("load command config must be a string-keyed mapping")
    if checkpoint_digest is not None and not isinstance(checkpoint_digest, str):
        raise TypeError("load command checkpoint digest must be a string or None")
    if not isinstance(checkpoint_root, str):
        raise TypeError("load command checkpoint root must be a string")

    source_config = _config_from_mapping(
        cast("dict[str, object]", mapping_value),
        expected_family,
    )
    settings = source_config.distributed
    if settings is None or settings.world_size <= 1:
        raise ValueError("rank workers require a multi-rank distributed configuration")
    local_config = config_for_rank(source_config, rank)
    _set_rank_local_cuda_device(local_config)
    rank_context = bootstrap_rank_context(settings, rank)
    try:
        trainer = build_trainer(
            source_config,
            trainer_type=trainer_type,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
            rank_context=rank_context,
        )
        if checkpoint_digest is not None:
            state = ContentAddressedCheckpointStore(Path(checkpoint_root)).load(checkpoint_digest)
            trainer.restore(
                adapter=state.adapter,
                optimizer=state.optimizer,
                rng=state.rng,
                step_cursor=state.step_cursor,
                data_cursor=state.data_cursor,
                loss=state.loss,
            )
        return trainer, rank_context
    except BaseException:
        rank_context.close()
        raise


def _worker_main(
    rank: int,
    connection: Connection,
    expected_family: str,
    trainer_type: _TrainerType,
    model_factory: _ModelFactory,
    data_source_factory: _DataSourceFactory,
) -> None:
    trainer: _Trainer | None = None
    rank_context: TorchDistributedRankContext | None = None
    try:
        while True:
            try:
                message = connection.recv()
            except EOFError:
                break
            if not isinstance(message, tuple) or len(message) != 2:
                command = "unknown"
                payload: object = None
            else:
                command_value, payload = message
                command = command_value if isinstance(command_value, str) else "unknown"
            try:
                result: object
                if command == "load":
                    if trainer is not None:
                        raise RuntimeError("rank worker is already loaded")
                    trainer, rank_context = _load_worker_trainer(
                        rank,
                        payload,
                        expected_family=expected_family,
                        trainer_type=trainer_type,
                        model_factory=model_factory,
                        data_source_factory=data_source_factory,
                    )
                    result = None
                elif command == "step":
                    if trainer is None:
                        raise RuntimeError("rank worker is not loaded")
                    result = trainer.train_step()
                elif command == "rng":
                    if trainer is None:
                        raise RuntimeError("rank worker is not loaded")
                    result = trainer.rng_state_dict()
                elif command == "sync-digest":
                    if trainer is None:
                        raise RuntimeError("rank worker is not loaded")
                    result = adapter_sync_digest(trainer)
                elif command == "teardown":
                    if rank_context is not None:
                        rank_context.close()
                        rank_context = None
                    trainer = None
                    connection.send(("ok", command, None))
                    break
                else:
                    raise ValueError(f"unknown rank worker command {command!r}")
                connection.send(("ok", command, result))
            except BaseException as exc:
                try:
                    connection.send(
                        (
                            "error",
                            command,
                            type(exc).__name__,
                            str(exc),
                            traceback.format_exc(),
                        )
                    )
                except (BrokenPipeError, EOFError, OSError):
                    pass
                break
    finally:
        if rank_context is not None:
            rank_context.close()
        connection.close()


@dataclass
class _WorkerHandle:
    rank: int
    process: BaseProcess
    connection: Connection


class RankWorkerGroup:
    """Persistent spawned command loops for ranks other than rank zero."""

    def __init__(
        self,
        world_size: int,
        *,
        expected_family: str,
        trainer_type: _TrainerType,
        model_factory: _ModelFactory,
        data_source_factory: _DataSourceFactory,
    ) -> None:
        if world_size <= 1:
            raise ValueError("rank worker groups require world size greater than one")
        try:
            ForkingPickler.dumps((trainer_type, model_factory, data_source_factory))
        except Exception as exc:
            raise ValueError("multi-rank training factories must be spawn-picklable") from exc

        self._closed = False
        self._teardown_result: bool | None = None
        self._workers: list[_WorkerHandle] = []
        context = torch.multiprocessing.get_context("spawn")
        try:
            for rank in range(1, world_size):
                parent_connection, child_connection = context.Pipe(duplex=True)
                try:
                    process = context.Process(
                        target=_worker_main,
                        args=(
                            rank,
                            child_connection,
                            expected_family,
                            trainer_type,
                            model_factory,
                            data_source_factory,
                        ),
                        name=f"dinkster-training-rank-{rank}",
                    )
                    process.start()
                except BaseException:
                    parent_connection.close()
                    raise
                finally:
                    child_connection.close()
                self._workers.append(_WorkerHandle(rank, process, parent_connection))
        except BaseException:
            self.close()
            raise

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(worker.rank for worker in self._workers)

    def _send(self, command: str, payload: object = None) -> None:
        if self._closed:
            raise RankWorkerError("rank worker group is closed")
        for worker in self._workers:
            if not worker.process.is_alive():
                raise RankWorkerError(
                    f"rank {worker.rank} exited before {command!r}"
                    f" with code {worker.process.exitcode}"
                )
            try:
                worker.connection.send((command, payload))
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise RankWorkerError(f"rank {worker.rank} could not receive {command!r}") from exc

    @staticmethod
    def _receive(worker: _WorkerHandle, command: str) -> object:
        while not worker.connection.poll(0.1):
            if not worker.process.is_alive():
                raise RankWorkerError(
                    f"rank {worker.rank} exited during {command!r}"
                    f" with code {worker.process.exitcode}"
                )
        try:
            response = worker.connection.recv()
        except (EOFError, OSError) as exc:
            raise RankWorkerError(
                f"rank {worker.rank} closed its connection during {command!r}"
            ) from exc
        if not isinstance(response, tuple) or len(response) < 3:
            raise RankWorkerError(f"rank {worker.rank} returned a malformed response")
        if response[0] == "ok" and response[1] == command and len(response) == 3:
            return response[2]
        if response[0] == "error" and response[1] == command and len(response) == 5:
            raise RankWorkerError(
                f"rank {worker.rank} failed {command!r}: {response[2]}: {response[3]}\n"
                f"{response[4]}"
            )
        raise RankWorkerError(f"rank {worker.rank} returned an unexpected response to {command!r}")

    def _receive_all(self, command: str) -> tuple[object, ...]:
        return tuple(self._receive(worker, command) for worker in self._workers)

    def start_load(
        self,
        config: dict[str, object],
        checkpoint_digest: str | None,
        checkpoint_root: Path,
    ) -> None:
        self._send("load", (config, checkpoint_digest, str(checkpoint_root)))

    def finish_load(self) -> None:
        self._receive_all("load")

    def start_step(self) -> None:
        self._send("step")

    def finish_step(self) -> tuple[float, ...]:
        values = self._receive_all("step")
        if not all(type(value) in (int, float) for value in values):
            raise RankWorkerError("rank worker returned a non-numeric loss")
        return tuple(float(cast("int | float", value)) for value in values)

    def rng_state_dicts(self) -> tuple[dict[str, torch.Tensor], ...]:
        self._send("rng")
        values = self._receive_all("rng")
        states: list[dict[str, torch.Tensor]] = []
        for value in values:
            if not isinstance(value, dict) or not all(
                isinstance(key, str) and isinstance(item, torch.Tensor)
                for key, item in value.items()
            ):
                raise RankWorkerError("rank worker returned an invalid RNG state")
            states.append(cast("dict[str, torch.Tensor]", value))
        return tuple(states)

    def sync_digests(self) -> tuple[str, ...]:
        self._send("sync-digest")
        values = self._receive_all("sync-digest")
        if not all(isinstance(value, str) for value in values):
            raise RankWorkerError("rank worker returned an invalid sync digest")
        return cast("tuple[str, ...]", values)

    def signal_teardown(self) -> bool:
        """Tell live workers to tear down without waiting for acknowledgments."""
        if self._teardown_result is not None:
            return self._teardown_result
        all_signaled = True
        for worker in self._workers:
            if not worker.process.is_alive():
                all_signaled = False
                continue
            try:
                worker.connection.send(("teardown", None))
            except (BrokenPipeError, EOFError, OSError):
                all_signaled = False
        self._teardown_result = all_signaled
        return all_signaled

    def close(self) -> None:
        if self._closed:
            return
        self.signal_teardown()
        self._closed = True

        deadline = time.monotonic() + 5.0
        for worker in self._workers:
            remaining = max(0.0, deadline - time.monotonic())
            if worker.process.is_alive() and worker.connection.poll(remaining):
                try:
                    worker.connection.recv()
                except (EOFError, OSError):
                    pass
            worker.process.join(timeout=max(0.0, deadline - time.monotonic()))
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=2.0)
            if worker.process.is_alive():
                worker.process.kill()
                worker.process.join(timeout=2.0)
            worker.connection.close()

    def __enter__(self) -> RankWorkerGroup:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
