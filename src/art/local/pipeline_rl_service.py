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
from typing import TYPE_CHECKING, AsyncIterator, cast

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
    def state(self) -> UnslothState:
        """
        Initialize Unsloth model for training.

        This is similar to DecoupledUnslothService._state but without any
        sleep/wake logic. The model is initialized once and kept in memory.

        IMPORTANT: This should be called AFTER setting CUDA_VISIBLE_DEVICES
        to ensure the trainer runs on the correct GPU (typically GPU 0).
        """
        import unsloth

        logger.info("=" * 80)
        logger.info("[PIPELINE_RL_SERVICE] Initializing trainer state")
        logger.info("=" * 80)
        logger.info(f"[PIPELINE_RL_SERVICE] Model: {self.model_name}")
        logger.info(f"[PIPELINE_RL_SERVICE] Base model: {self.base_model}")
        logger.info(f"[PIPELINE_RL_SERVICE] Output dir: {self.output_dir}")
        logger.info("")

        # Get trainer GPU IDs from config (default to [0])
        trainer_gpu_ids = self.config.get("trainer_gpu_ids", [0])
        if not trainer_gpu_ids:
            trainer_gpu_ids = [0]

        logger.info(
            f"[PIPELINE_RL_SERVICE] Trainer GPU configuration: {trainer_gpu_ids}"
        )

        # Set CUDA_VISIBLE_DEVICES for trainer before initializing model
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        trainer_gpu_str = ",".join(str(gpu_id) for gpu_id in trainer_gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = trainer_gpu_str
        logger.info(
            f"[PIPELINE_RL_SERVICE] Setting CUDA_VISIBLE_DEVICES={trainer_gpu_str} for trainer"
        )
        logger.info(f"[PIPELINE_RL_SERVICE]   (original was: {original_cuda_visible})")
        logger.info("")

        # Initialize Unsloth model
        logger.info("[PIPELINE_RL_SERVICE] Step 1: Loading model...")
        init_args = self.config.get("init_args", {})
        checkpoint_dir = get_last_checkpoint_dir(self.output_dir)
        if checkpoint_dir:
            logger.info(
                f"[PIPELINE_RL_SERVICE]   Loading from checkpoint: {checkpoint_dir}"
            )
            init_args["model_name"] = checkpoint_dir
        else:
            logger.info(
                f"[PIPELINE_RL_SERVICE]   Loading base model: {self.base_model}"
            )
            init_args["model_name"] = self.base_model

        logger.info(f"[PIPELINE_RL_SERVICE]   Process PID: {os.getpid()}")
        logger.info(
            f"[PIPELINE_RL_SERVICE]   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )

        model, tokenizer = cast(
            tuple[CausalLM, PreTrainedTokenizerBase],
            unsloth.FastLanguageModel.from_pretrained(**init_args),
        )

        logger.info(f"[PIPELINE_RL_SERVICE]   Model loaded on device: {model.device}")
        logger.info("")

        # Initialize PEFT model
        logger.info("[PIPELINE_RL_SERVICE] Step 2: Initializing PEFT model...")
        peft_model = cast(
            peft.peft_model.PeftModelForCausalLM,
            unsloth.FastLanguageModel.get_peft_model(
                model, **self.config.get("peft_args", {})
            ),
        )
        logger.info(f"[PIPELINE_RL_SERVICE]   PEFT model initialized")
        logger.info("")

        # Initialize trainer with dummy dataset
        logger.info("[PIPELINE_RL_SERVICE] Step 3: Initializing GRPOTrainer...")
        data = {"prompt": ""}
        trainer = GRPOTrainer(
            model=peft_model,  # type: ignore
            reward_funcs=[],
            args=GRPOConfig(**self.config.get("trainer_args", {})),  # type: ignore
            train_dataset=Dataset.from_list([data for _ in range(10_000_000)]),
            processing_class=tokenizer,
        )

        logger.info(f"[PIPELINE_RL_SERVICE]   Trainer device: {trainer.args.device}")
        logger.info(
            f"[PIPELINE_RL_SERVICE]   Trainer accelerator device: {trainer.accelerator.device}"
        )
        logger.info("")

        # Initialize queues
        logger.info("[PIPELINE_RL_SERVICE] Step 4: Initializing queues...")
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
        logger.info("[PIPELINE_RL_SERVICE]   Queues initialized")
        logger.info("")

        logger.info("[PIPELINE_RL_SERVICE] Trainer state initialization complete!")
        logger.info(
            f"[PIPELINE_RL_SERVICE] CUDA_VISIBLE_DEVICES remains set to {trainer_gpu_str} for trainer process"
        )
        logger.info("=" * 80)
        logger.info("")

        return UnslothState(
            model=model,
            tokenizer=tokenizer,
            peft_model=peft_model,
            trainer=trainer,
            inputs_queue=inputs_queue,
            results_queue=results_queue,
        )

    async def initialize_process_groups_and_vllm(
        self, config: dev.OpenAIServerConfig | None
    ):
        logger.info("[PIPELINE_RL_SERVICE] Initializing state...")
        _ = self.state
        # Initialize process group
        logger.info("[PIPELINE_RL_SERVICE] Initializing process group...")
        init_method, world_size = await self._init_process_group_for_weight_updates(
            len(self.config.get("inference_gpu_ids", [1]))
        )
        logger.info(
            f"[PIPELINE_RL_SERVICE] Process group initialized: {init_method}, {world_size}"
        )

        # Start vLLM server
        logger.info("[PIPELINE_RL_SERVICE] Starting vLLM server...")
        await self.start_openai_server_with_weight_updates(
            config=config,
            init_method=init_method,
            world_size=world_size,
            actor_idx=0,
            inference_gpu_ids=self.config.get("inference_gpu_ids", [1]),
        )
        logger.info("[PIPELINE_RL_SERVICE] vLLM server started")

        # Join process group
        logger.info("[PIPELINE_RL_SERVICE] Joining process group...")
        self._actor_update_group = await self._join_process_group_as_trainer(
            init_method=init_method,
            world_size=world_size,
        )
        logger.info("[PIPELINE_RL_SERVICE] Process group joined")

        # Wait for vLLM server to be ready
        logger.info("[PIPELINE_RL_SERVICE] Waiting for vLLM server to be ready...")
        server_config = config or dev.OpenAIServerConfig()
        server_args = server_config.get("server_args", {})
        base_url = f"http://{server_args.get('host', '0.0.0.0')}:{server_args.get('port', 8000)}/v1"
        await self._wait_for_vllm_ready(base_url)
        logger.info("[PIPELINE_RL_SERVICE] vLLM server is ready")

    async def start_openai_server(self, config: dev.OpenAIServerConfig | None) -> None:
        """
        Skip this function, vLLM server is initiated to `initialize_process_groups_and_vllm`
        """
        pass

    async def stop_openai_server(self) -> None:
        """
        TODO
        """
        pass

    async def start_openai_server_with_weight_updates(
        self,
        config: dev.OpenAIServerConfig | None,
        init_method: str,
        world_size: int,
        actor_idx: int = 0,
        inference_gpu_ids: list[int] | None = None,
    ) -> None:
        """
        Start vLLM with weight update support for PipelineRL.

        Args:
            config: OpenAI server configuration
            init_method: TCP endpoint for process group (e.g., "tcp://localhost:12345")
            world_size: Total processes in weight update group (trainer + vLLM GPUs)
            actor_idx: Index of this actor (for multi-actor setups, default 0)
            inference_gpu_ids: List of GPU IDs for vLLM inference (default: [0])
        """
        if inference_gpu_ids is None:
            inference_gpu_ids = [0]

        logger.info("[PIPELINE_RL_SERVICE] Starting vLLM with weight update support")
        logger.info(f"[PIPELINE_RL_SERVICE]   init_method: {init_method}")
        logger.info(f"[PIPELINE_RL_SERVICE]   world_size: {world_size}")
        logger.info(f"[PIPELINE_RL_SERVICE]   actor_idx: {actor_idx}")
        logger.info(f"[PIPELINE_RL_SERVICE]   inference_gpu_ids: {inference_gpu_ids}")

        # Get the path to the custom vllm_server.py script
        import art

        art_root = os.path.dirname(os.path.dirname(os.path.dirname(art.__file__)))
        vllm_server_script = os.path.join(art_root, "dev/pipeline-rl/vllm_server.py")

        if not os.path.exists(vllm_server_script):
            raise FileNotFoundError(
                f"vllm_server.py not found at {vllm_server_script}. "
                "Make sure dev/pipeline-rl/vllm_server.py exists in the ART repository."
            )

        logger.info(
            f"[PIPELINE_RL_SERVICE]   Using vllm_server.py at: {vllm_server_script}"
        )

        # Get server configuration
        server_config = config or {}
        server_args = server_config.get("server_args", {})
        host = server_args.get("host", "0.0.0.0")
        port = server_args.get("port", 8000)

        # Determine model path (use last checkpoint if exists, otherwise base model)
        from ..local.checkpoints import get_last_checkpoint_dir

        checkpoint_dir = get_last_checkpoint_dir(self.output_dir)
        if checkpoint_dir:
            logger.info(
                f"[PIPELINE_RL_SERVICE]   Loading from checkpoint: {checkpoint_dir}"
            )
            model_path = checkpoint_dir
        else:
            logger.info(
                f"[PIPELINE_RL_SERVICE]   Loading base model: {self.base_model}"
            )
            model_path = self.base_model

        # Build command to launch vllm_server.py
        # We use --served-model-name so vLLM serves the model under the TrainableModel's name
        # This allows rollout() to call the API with model.get_inference_name() (e.g., "agent-001")
        logger.info(
            f"[PIPELINE_RL_SERVICE]   vLLM will serve model '{model_path}' as '{self.model_name}'"
        )
        cmd = [
            "uv",
            "run",
            vllm_server_script,
            "--model",
            model_path,
            "--served-model-name",
            self.model_name,  # Serve under the TrainableModel's name (e.g., "agent-001")
            "--host",
            host,
            "--port",
            str(port),
            "--weight-update-group-init-method",
            init_method,
            "--weight-update-group-world-size",
            str(world_size),
            "--actor-llm-idx",
            str(actor_idx),
            "--gpu-memory-utilization",
            str(0.8),
            "--max-model-len",
            str(32768),
            "--full-cuda-graph",
            "--cudagraph-num-of-warmups",
            str(1),
            "--cudagraph-capture-sizes",
            "400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1",
            "--cudagraph-max-capture-size",
            "400",
            "--enable-prefix-caching",
            "--enable-lora",
        ]

        # Add optional server args from config
        # if "max_model_len" in server_args:
        #     cmd.extend(["--max-model-len", str(server_args["max_model_len"])])
        # if "gpu_memory_utilization" in server_args:
        #     cmd.extend(
        #         ["--gpu-memory-utilization", str(server_args["gpu_memory_utilization"])]
        #     )
        if "tensor_parallel_size" in server_args:
            cmd.extend(
                ["--tensor-parallel-size", str(server_args["tensor_parallel_size"])]
            )

        # Add LoRA configuration from config (optional, with defaults)
        max_loras = server_args.get("max_loras", 1)
        max_lora_rank = server_args.get("max_lora_rank", 8)
        cmd.extend(["--max-loras", str(max_loras)])
        cmd.extend(["--max-lora-rank", str(max_lora_rank)])

        # Add log directory if specified
        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        cmd.extend(["--log-dir", log_dir])

        logger.info(f"[PIPELINE_RL_SERVICE] Launching vLLM with command:")
        logger.info(f"[PIPELINE_RL_SERVICE]   {' '.join(cmd)}")

        # Prepare environment for vLLM subprocess
        # We need to control which GPUs vLLM sees via CUDA_VISIBLE_DEVICES
        vllm_env = os.environ.copy()

        # Set CUDA_VISIBLE_DEVICES to only show inference GPUs for vLLM
        inference_gpu_str = ",".join(str(gpu_id) for gpu_id in inference_gpu_ids)
        vllm_env["CUDA_VISIBLE_DEVICES"] = inference_gpu_str
        logger.info(
            f"[PIPELINE_RL_SERVICE] Setting CUDA_VISIBLE_DEVICES={inference_gpu_str} for vLLM subprocess"
        )

        # Enable runtime LoRA updating (required for Phase 1 LoRA swapping)
        vllm_env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        logger.info(
            f"[PIPELINE_RL_SERVICE] Setting VLLM_ALLOW_RUNTIME_LORA_UPDATING=True"
        )

        # Start the vLLM server process
        # Note: This will block during startup until the trainer joins the process group
        self._vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=vllm_env,
        )

        logger.info(
            f"[PIPELINE_RL_SERVICE] vLLM process started (PID: {self._vllm_process.pid})"
        )
        logger.info(
            "[PIPELINE_RL_SERVICE] vLLM will block during startup until trainer joins process group"
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
        # Import torch here (already imported in state, but needed in this scope)
        import torch

        logger.info(
            f"[PIPELINE_RL_SERVICE] Starting training step {self._training_step}"
        )

        # Load packed tensors
        packed_tensors = packed_tensors_from_dir(**disk_packed_tensors)

        # Wait for existing batches to finish
        await self.state.results_queue.join()

        # If we haven't already, start the training task
        if self._train_task is None:
            logger.info("[PIPELINE_RL_SERVICE] Starting background training task")
            self._train_task = asyncio.create_task(
                train(
                    trainer=self.state.trainer,
                    results_queue=self.state.results_queue,
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
                            self.state.trainer.compute_loss(
                                self.state.peft_model,
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

                self.state.inputs_queue.put_nowait(
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
                        asyncio.create_task(self.state.results_queue.get()),
                        self._train_task,
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if verbose:
                    logger.debug("[PIPELINE_RL_SERVICE] Received training result")

                for task in done:
                    result = task.result()
                    assert result is not None, "The training task should never finish."
                    self.state.results_queue.task_done()

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
        self.state.trainer.save_model(checkpoint_dir)

        logger.info(f"[PIPELINE_RL_SERVICE] Saved checkpoint to {checkpoint_dir}")
        logger.info(
            f"[PIPELINE_RL_SERVICE] Completed training step {self._training_step}"
        )

    async def swap_lora_checkpoint(
        self, checkpoint_dir: str, base_url: str = "http://localhost:8000"
    ) -> None:
        """
        Swap LoRA adapter after training (Phase 1 approach).

        This uses the existing vLLM LoRA swapping mechanism via HTTP API,
        which briefly pauses generation to load the new checkpoint.

        Args:
            checkpoint_dir: Path to the new LoRA checkpoint
            base_url: Base URL of the vLLM server
        """
        logger.info(f"[LORA_SWAP] Swapping to checkpoint: {checkpoint_dir}")
        logger.info(f"[LORA_SWAP] vLLM server URL: {base_url}")

        import aiohttp

        # vLLM's LoRA management endpoints
        # See: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#lora-adapters
        load_lora_url = f"{base_url}/v1/load_lora_adapter"
        unload_lora_url = f"{base_url}/v1/unload_lora_adapter"

        async with aiohttp.ClientSession() as session:
            # Step 1: Unload old LoRA (adapter name + ID)
            logger.info(
                f"[LORA_SWAP] Step 1: Unloading old LoRA adapter '{self.model_name}' (ID: 1)"
            )
            try:
                async with session.post(
                    unload_lora_url,
                    json={
                        "lora_name": self.model_name,  # Required by vLLM API
                        "lora_int_id": 1,
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    response_text = await response.text()
                    logger.info(
                        f"[LORA_SWAP]   Unload response status: {response.status}"
                    )
                    logger.info(f"[LORA_SWAP]   Unload response text: {response_text}")

                    if response.status == 200:
                        logger.info("[LORA_SWAP]   Old LoRA unloaded successfully")
                    else:
                        # It's OK if the LoRA doesn't exist (first time)
                        if (
                            "not found" in response_text.lower()
                            or "not loaded" in response_text.lower()
                        ):
                            logger.info(
                                "[LORA_SWAP]   No existing LoRA to unload (first swap)"
                            )
                        else:
                            logger.warning(
                                f"[LORA_SWAP]   Failed to unload LoRA (non-fatal): {response_text}"
                            )
            except Exception as e:
                logger.warning(
                    f"[LORA_SWAP]   Error unloading LoRA (continuing anyway): {e}"
                )

            # Step 2: Load new LoRA
            logger.info(f"[LORA_SWAP] Step 2: Loading new LoRA from {checkpoint_dir}")

            # Verify checkpoint directory exists and has required files
            logger.info(f"[LORA_SWAP]   Checking checkpoint directory...")
            if not os.path.exists(checkpoint_dir):
                raise RuntimeError(
                    f"Checkpoint directory does not exist: {checkpoint_dir}"
                )

            checkpoint_files = os.listdir(checkpoint_dir)
            logger.info(
                f"[LORA_SWAP]   Checkpoint directory contents: {checkpoint_files}"
            )

            # Check for required LoRA files
            has_adapter_config = "adapter_config.json" in checkpoint_files
            has_adapter_model = any(
                f.startswith("adapter_model") for f in checkpoint_files
            )
            logger.info(f"[LORA_SWAP]   Has adapter_config.json: {has_adapter_config}")
            logger.info(f"[LORA_SWAP]   Has adapter_model files: {has_adapter_model}")

            if not has_adapter_config or not has_adapter_model:
                logger.warning(
                    f"[LORA_SWAP]   Checkpoint directory missing required LoRA files!"
                )

            # Prepare request payload
            payload = {
                "lora_name": self.model_name,
                "lora_int_id": 1,
                "lora_path": checkpoint_dir,
            }
            logger.info(f"[LORA_SWAP]   Request URL: {load_lora_url}")
            logger.info(f"[LORA_SWAP]   Request payload: {payload}")

            try:
                async with session.post(
                    load_lora_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    response_text = await response.text()
                    logger.info(f"[LORA_SWAP]   Response status: {response.status}")
                    logger.info(f"[LORA_SWAP]   Response text: {response_text}")

                    if response.status == 200:
                        logger.info("[LORA_SWAP]   New LoRA loaded successfully")
                    else:
                        raise RuntimeError(
                            f"Failed to load new LoRA (status {response.status}): {response_text}"
                        )
            except Exception as e:
                logger.error(f"[LORA_SWAP]   Failed to load new LoRA: {e}")
                raise

        logger.info(f"[LORA_SWAP] Swap complete!")

    async def _set_lora(
        self, lora_path: str, base_url: str = "http://localhost:8000"
    ) -> None:
        """
        Set the LoRA adapter in vLLM via HTTP API.

        Args:
            lora_path: Path to the LoRA checkpoint directory
            base_url: Base URL of the vLLM server
        """
        logger.info(f"[SET_LORA] Setting LoRA adapter: {lora_path}")
        logger.info(f"[SET_LORA] vLLM server URL: {base_url}")

        import aiohttp

        # vLLM's LoRA management endpoint
        load_lora_url = f"{base_url}/v1/load_lora_adapter"

        async with aiohttp.ClientSession() as session:
            # Prepare request payload
            payload = {
                "lora_name": self.model_name,
                "lora_int_id": 1,
                "lora_path": lora_path,
            }
            logger.info(f"[SET_LORA]   Request URL: {load_lora_url}")
            logger.info(f"[SET_LORA]   Request payload: {payload}")

            try:
                async with session.post(
                    load_lora_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    response_text = await response.text()
                    logger.info(f"[SET_LORA]   Response status: {response.status}")
                    logger.info(f"[SET_LORA]   Response text: {response_text}")

                    if response.status == 200:
                        logger.info("[SET_LORA]   LoRA adapter loaded successfully")
                    else:
                        raise RuntimeError(
                            f"Failed to load LoRA adapter (status {response.status}): {response_text}"
                        )
            except Exception as e:
                logger.error(f"[SET_LORA]   Failed to load LoRA adapter: {e}")
                raise

    def _get_free_port(self) -> int:
        """
        Find a free TCP port for process group initialization.

        Returns:
            int: A free port number
        """
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    async def _init_process_group_for_weight_updates(
        self,
        num_actor_gpus: int,
    ) -> tuple[str, int]:
        """
        Initialize process group for weight updates.

        This creates the TCP rendezvous endpoint that will be used by both the trainer
        and vLLM to coordinate weight updates via NCCL.

        Args:
            num_actor_gpus: Number of GPUs dedicated to vLLM generation

        Returns:
            tuple: (init_method, world_size) for process group initialization
        """
        # Get free port for TCP rendezvous
        port = self._get_free_port()
        init_method = f"tcp://localhost:{port}"
        world_size = 1 + num_actor_gpus  # trainer (1) + actor GPUs

        logger.info("[INIT_PG] Step 1: Creating TCP endpoint")
        logger.info(f"[INIT_PG]   init_method: {init_method}")
        logger.info(f"[INIT_PG]   world_size: {world_size}")
        logger.info(f"[INIT_PG]   num_actor_gpus: {num_actor_gpus}")

        # Return init_method for vLLM to use
        # Trainer will join after vLLM starts
        return init_method, world_size

    async def _join_process_group_as_trainer(
        self,
        init_method: str,
        world_size: int,
    ) -> torch.distributed.ProcessGroup:
        """
        Join process group as trainer (rank 0).

        This should be called AFTER vLLM has started and is waiting to join
        the process group. The vLLM process will block until this method
        is called.

        Args:
            init_method: TCP endpoint (e.g., "tcp://localhost:12345")
            world_size: Total processes in group

        Returns:
            ProcessGroup: The initialized NCCL process group
        """
        from .torch_utils import init_extra_process_group

        logger.info("[INIT_PG] Step 4: Trainer joining process group as rank 0")
        logger.info(f"[INIT_PG]   This will BLOCK until vLLM joins...")

        pg = init_extra_process_group(
            backend="nccl",
            init_method=init_method,
            rank=0,
            world_size=world_size,
            group_name="actor",
        )

        logger.info("[INIT_PG] Step 5: Process group initialized successfully!")
        return pg

    async def _wait_for_vllm_ready(self, base_url: str, timeout: float = 120.0) -> None:
        """
        Wait for vLLM server to be ready by polling the /v1/models endpoint.

        Args:
            base_url: Base URL of the vLLM server (e.g., "http://localhost:8000")
            timeout: Maximum time to wait in seconds (default: 120)

        Raises:
            TimeoutError: If server doesn't become ready within timeout
        """
        logger.info(f"[WAIT_VLLM] Waiting for vLLM server at {base_url} to be ready...")
        logger.info(f"[WAIT_VLLM]   Timeout: {timeout}s")

        import time

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"vLLM server at {base_url} did not become ready within {timeout}s. "
                    f"Check the vLLM logs in the model output directory."
                )

            try:
                # Try to list models - this will succeed when vLLM is ready
                logger.info(
                    f"[WAIT_VLLM]   Attempting to list models (elapsed: {elapsed:.1f}s)..."
                )
                models = await client.models.list()

                # Try to iterate and get model IDs
                model_ids = []
                try:
                    # models might be a Page object or SyncPage, need to handle iteration carefully
                    async for model in models:
                        model_ids.append(model.id)
                except TypeError:
                    # Not async iterable, try regular iteration
                    for model in models:
                        model_ids.append(model.id)

                # If we successfully got models, vLLM is ready
                logger.info(f"[WAIT_VLLM] vLLM is ready! Available models: {model_ids}")
                return
            except Exception as e:
                # vLLM not ready yet, wait and retry
                logger.info(
                    f"[WAIT_VLLM]   Not ready yet ({elapsed:.1f}s): {type(e).__name__}: {e}"
                )
                await asyncio.sleep(0.5)
