"""
PipelineRL Service for concurrent generation and training.

This service manages vLLM and Unsloth for PipelineRL without the sleep/wake
mechanism used in DecoupledUnslothService. It keeps vLLM continuously generating
while Unsloth trains on separate GPUs.
"""

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass
from functools import cached_property
from typing import AsyncIterator, cast

import peft
import torch
from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.dummy_pt_objects import GenerationMixin, PreTrainedModel
from trl import GRPOConfig, GRPOTrainer

from .. import dev, types
from ..local.checkpoints import get_last_checkpoint_dir
from ..preprocessing.pack import (
    DiskPackedTensors,
    PackedTensors,
    packed_tensors_from_dir,
)
from ..unsloth.train import train
from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir

logger = logging.getLogger(__name__)


# copied from src/art/unsloth/decoupled_service.py
class CausalLM(PreTrainedModel, GenerationMixin):
    """Dummy class for type checking."""

    pass

# copied from src/art/unsloth/decoupled_service.py
class TrainInputs(PackedTensors):
    """Training inputs extending PackedTensors with config."""
    config: types.TrainConfig
    _config: dev.TrainConfig
    return_new_logprobs: bool

# copied from src/art/unsloth/decoupled_service.py
@dataclass
class UnslothState:
    """State for Unsloth training."""
    model: CausalLM
    tokenizer: PreTrainedTokenizerBase
    peft_model: peft.peft_model.PeftModelForCausalLM
    trainer: GRPOTrainer
    inputs_queue: asyncio.Queue[TrainInputs]
    results_queue: asyncio.Queue[dict[str, float]]

@dataclass
class PipelineRLService:
    """
    Service for PipelineRL with separate vLLM and training processes.

    Unlike DecoupledUnslothService, this service does NOT put vLLM to sleep
    during training. Instead, vLLM continues generating on separate GPUs while
    Unsloth trains, with periodic LoRA checkpoint swaps.

    Attributes:
        model_name: Name of the model
        base_model: Base model identifier (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        config: Internal model configuration
        output_dir: Directory for outputs and checkpoints
        _vllm_process: Subprocess running vLLM server
        _generation_step: Current generation step (for tracking)
        _training_step: Current training step (for tracking)
        _train_task: Background training task
    """

    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str
    _vllm_process: subprocess.Popen | None = None
    _generation_step: int = 0
    _training_step: int = 0
    _train_task: asyncio.Task[None] | None = None

    @cached_property
    def _state(self) -> UnslothState:
        """
        Initialize Unsloth model for training.

        This is similar to DecoupledUnslothService._state but without any
        sleep/wake logic. The model is initialized once and kept in memory.
        """
        logger.info("[PIPELINE_RL_SERVICE] Initializing Unsloth state...")

        import unsloth

        # Initialize Unsloth model
        init_args = self.config.get("init_args", {})
        checkpoint_dir = get_last_checkpoint_dir(self.output_dir)
        if checkpoint_dir:
            logger.info(f"[PIPELINE_RL_SERVICE] Loading from checkpoint: {checkpoint_dir}")
            init_args["model_name"] = checkpoint_dir
        else:
            logger.info(f"[PIPELINE_RL_SERVICE] Loading base model: {self.base_model}")
            init_args["model_name"] = self.base_model

        model, tokenizer = cast(
            tuple[CausalLM, PreTrainedTokenizerBase],
            unsloth.FastLanguageModel.from_pretrained(**init_args),
        )

        # Initialize PEFT model
        peft_model = cast(
            peft.peft_model.PeftModelForCausalLM,
            unsloth.FastLanguageModel.get_peft_model(
                model, **self.config.get("peft_args", {})
            ),
        )

        # Initialize trainer with dummy dataset
        data = {"prompt": ""}
        trainer = GRPOTrainer(
            model=peft_model,  # type: ignore
            reward_funcs=[],
            args=GRPOConfig(**self.config.get("trainer_args", {})),  # type: ignore
            train_dataset=Dataset.from_list([data for _ in range(10_000_000)]),
            processing_class=tokenizer,
        )

        # Initialize queues
        inputs_queue: asyncio.Queue[TrainInputs] = asyncio.Queue()
        results_queue: asyncio.Queue[dict[str, float]] = asyncio.Queue()

        # Patch trainer _prepare_inputs() to pull from queue
        def _async_prepare_inputs(*_: any, **__: any) -> dict[str, torch.Tensor]:
            async def get_inputs() -> TrainInputs:
                return await inputs_queue.get()

            # Force otherwise synchronous _prepare_inputs() to yield
            # with nested asyncio.run() call
            inputs = asyncio.run(get_inputs())

            return cast(dict[str, torch.Tensor], inputs)

        trainer._prepare_inputs = _async_prepare_inputs

        logger.info("[PIPELINE_RL_SERVICE] Unsloth state initialized")

        return UnslothState(
            model=model,
            tokenizer=tokenizer,
            peft_model=peft_model,
            trainer=trainer,
            inputs_queue=inputs_queue,
            results_queue=results_queue,
        )

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> None:
        """
        Start vLLM OpenAI-compatible server (standard mode, no weight updates yet).

        This is the basic version used for conventional RL. For PipelineRL with
        weight updates, use start_openai_server_with_weight_updates().
        """
        logger.info("[PIPELINE_RL_SERVICE] Starting vLLM server (standard mode)")
        # TODO: Implement standard vLLM startup
        # For now, raise NotImplementedError
        raise NotImplementedError(
            "start_openai_server not yet implemented for PipelineRLService. "
            "Use start_openai_server_with_weight_updates() for PipelineRL mode."
        )

    async def start_openai_server_with_weight_updates(
        self,
        config: dev.OpenAIServerConfig | None,
        init_method: str,
        world_size: int,
        actor_idx: int = 0,
    ) -> None:
        """
        Start vLLM with weight update support for PipelineRL.

        Args:
            config: OpenAI server configuration
            init_method: TCP endpoint for process group (e.g., "tcp://localhost:12345")
            world_size: Total processes in weight update group (trainer + vLLM GPUs)
            actor_idx: Index of this actor (for multi-actor setups, default 0)
        """
        logger.info("[PIPELINE_RL_SERVICE] Starting vLLM with weight update support")
        logger.info(f"[PIPELINE_RL_SERVICE]   init_method: {init_method}")
        logger.info(f"[PIPELINE_RL_SERVICE]   world_size: {world_size}")
        logger.info(f"[PIPELINE_RL_SERVICE]   actor_idx: {actor_idx}")

        # TODO: Implement vLLM startup with weight update support
        # This will launch the custom vllm_server.py script
        raise NotImplementedError(
            "start_openai_server_with_weight_updates not yet implemented"
        )

    async def vllm_engine_is_sleeping(self) -> bool:
        """
        Check if vLLM engine is sleeping.

        For PipelineRLService, vLLM never sleeps (unlike DecoupledUnslothService),
        so this always returns False.
        """
        return False

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """
        Train on packed tensors.

        This is similar to DecoupledUnslothService.train() but WITHOUT:
        - sleep_task for putting vLLM to sleep
        - vLLM sleep/wake calls
        - Memory cleanup between vLLM and training

        vLLM runs continuously on separate GPUs.
        """
        logger.info(f"[PIPELINE_RL_SERVICE] Starting training step {self._training_step}")

        # Load packed tensors
        packed_tensors = packed_tensors_from_dir(**disk_packed_tensors)

        # Wait for existing batches to finish
        await self._state.results_queue.join()

        # If we haven't already, start the training task
        if self._train_task is None:
            logger.info("[PIPELINE_RL_SERVICE] Starting background training task")
            self._train_task = asyncio.create_task(
                train(
                    trainer=self._state.trainer,
                    results_queue=self._state.results_queue,
                )
            )
            warmup = True
        else:
            warmup = False

        precalculate_logprobs = _config.get("precalculate_logprobs", False)

        # Train on the batch
        for offset in range(0, packed_tensors["tokens"].shape[0]):
            for _ in range(2 if warmup else 1):
                if precalculate_logprobs and not warmup:
                    # Precalculate logprobs if needed
                    packed_tensors["logprobs"] = torch.cat(
                        [
                            self._state.trainer.compute_loss(
                                self._state.peft_model,
                                TrainInputs(
                                    **{
                                        k: v[_offset : _offset + 1]
                                        for k, v in packed_tensors.items()
                                        if isinstance(v, torch.Tensor)
                                    },
                                    pixel_values=packed_tensors["pixel_values"][
                                        _offset : _offset + 1
                                    ],
                                    image_grid_thw=packed_tensors["image_grid_thw"][
                                        _offset : _offset + 1
                                    ],
                                    config=config,
                                    _config=_config,
                                    return_new_logprobs=True,
                                ),  # type: ignore
                            )
                            for _offset in range(0, packed_tensors["tokens"].shape[0])
                        ]
                    ).to("cpu")
                    precalculate_logprobs = False

                self._state.inputs_queue.put_nowait(
                    TrainInputs(
                        **{
                            k: (
                                v[offset : offset + 1, :1024]
                                if warmup and v.dim() > 1
                                else v[offset : offset + 1]
                            )
                            for k, v in packed_tensors.items()
                            if isinstance(v, torch.Tensor)
                        },
                        pixel_values=(
                            [None]
                            if warmup
                            else packed_tensors["pixel_values"][offset : offset + 1]
                        ),
                        image_grid_thw=(
                            [None]
                            if warmup
                            else packed_tensors["image_grid_thw"][offset : offset + 1]
                        ),
                        config=(
                            config.model_copy(
                                update={"lr": 1e-9, "beta": 0.0, "kl_coef": 0.0}
                            )
                            if warmup
                            else config
                        ),
                        _config=_config,
                        return_new_logprobs=False,
                    )
                )

                # Wait for a result from the queue or for the training task to raise an exception
                done, _ = await asyncio.wait(
                    [
                        asyncio.create_task(self._state.results_queue.get()),
                        self._train_task,
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if verbose:
                    logger.debug("[PIPELINE_RL_SERVICE] Received training result")

                for task in done:
                    result = task.result()
                    assert result is not None, "The training task should never finish."
                    self._state.results_queue.task_done()

                    if warmup:
                        from ..unsloth.train import gc_and_empty_cuda_cache

                        gc_and_empty_cuda_cache()
                        await asyncio.sleep(0.1)
                        warmup = False
                    else:
                        yield result

        if verbose:
            logger.info("[PIPELINE_RL_SERVICE] Saving new LoRA adapter...")

        # Save checkpoint after training
        next_step = get_step_from_dir(self.output_dir) + 1
        checkpoint_dir = get_step_checkpoint_dir(self.output_dir, next_step)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._state.trainer.save_model(checkpoint_dir)

        logger.info(f"[PIPELINE_RL_SERVICE] Saved checkpoint to {checkpoint_dir}")
        logger.info(f"[PIPELINE_RL_SERVICE] Completed training step {self._training_step}")

    async def swap_lora_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Swap LoRA adapter after training (Phase 1 approach).

        This uses the existing vLLM LoRA swapping mechanism, which briefly
        pauses generation to load the new checkpoint.

        Args:
            checkpoint_dir: Path to the new LoRA checkpoint
        """
        logger.info(f"[PIPELINE_RL_SERVICE] Swapping to checkpoint: {checkpoint_dir}")

        # TODO: Implement LoRA swapping via vLLM API
        # This will use vLLM's remove_lora() and add_lora() methods
        raise NotImplementedError("swap_lora_checkpoint not yet implemented")
