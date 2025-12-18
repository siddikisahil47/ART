import asyncio
import functools
import logging
import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, AsyncIterator, cast

import httpx
import unsloth
import peft
import torch
from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.dummy_pt_objects import GenerationMixin, PreTrainedModel
from trl import GRPOConfig, GRPOTrainer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from art.dev.openai_server import ServerArgs
from art.local.checkpoints import get_last_checkpoint_dir
from art.nccl import NCCLBroadcastSender
from art.utils.get_model_step import get_step_from_dir
from art.utils.output_dirs import get_step_checkpoint_dir

from .. import dev, types
from ..preprocessing.pack import (
    DiskPackedTensors,
    PackedTensors,
    packed_tensors_from_dir,
)
from .train import train

logger = logging.getLogger(__name__)


class CausalLM(PreTrainedModel, GenerationMixin):
    """Dummy class for type checking."""

    pass


class TrainInputs(PackedTensors):
    config: types.TrainConfig
    _config: dev.TrainConfig
    return_new_logprobs: bool


@dataclass
class StateDictWrapper:
    """Wrapper for a state dict to behave like a model for broadcasting."""

    _state_dict: dict[str, torch.Tensor]

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._state_dict


@dataclass
class AsyncState:
    model: CausalLM
    tokenizer: PreTrainedTokenizerBase
    peft_model: peft.peft_model.PeftModelForCausalLM
    trainer: GRPOTrainer
    inputs_queue: asyncio.Queue[TrainInputs]
    results_queue: asyncio.Queue[dict[str, float]]


@dataclass
class AsyncService:
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str
    _openai_server_task: asyncio.Task[None] | None = None
    _train_task: asyncio.Task[None] | None = None
    nccl_broadcast: NCCLBroadcastSender | None = None

    @functools.cached_property
    def state(self) -> AsyncState:
        # Initialize Unsloth model
        init_args = self.config.get("init_args", {})
        checkpoint_dir = get_last_checkpoint_dir(self.output_dir)
        if checkpoint_dir:
            init_args["model_name"] = checkpoint_dir
        else:
            init_args["model_name"] = self.base_model

        # NOTE: We have to patch empty_cache with a no-op during model initialization
        # to avoid an allocator error.
        empty_cache = torch.cuda.empty_cache
        torch.cuda.empty_cache = lambda: None
        from_engine_args = AsyncLLMEngine.from_engine_args

        # NOTE: We also have to patch from_engine_args to control the engine args
        # that are passed to the engine constructor.
        def _from_engine_args(
            engine_args: AsyncEngineArgs, *args: Any, **kwargs: Any
        ) -> AsyncLLMEngine:
            return from_engine_args(
                replace(engine_args, **self.config.get("engine_args", {})),
                *args,
                **kwargs,
            )

        AsyncLLMEngine.from_engine_args = _from_engine_args
        init_args["load_in_4bit"] = False
        logger.info(f"[ASYNC_SERVICE] DEBUG: Init args: {init_args}")
        model, tokenizer = cast(
            tuple[CausalLM, PreTrainedTokenizerBase],
            unsloth.FastLanguageModel.from_pretrained(**init_args),
        )
        AsyncLLMEngine.from_engine_args = from_engine_args
        torch.cuda.empty_cache = empty_cache
        torch.cuda.empty_cache()

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
        def _async_prepare_inputs(*_: Any, **__: Any) -> dict[str, torch.Tensor]:
            async def get_inputs() -> TrainInputs:
                return await inputs_queue.get()

            # Force otherwise synchronous _prepare_inputs() to yield
            # with nested asyncio.run() call
            inputs = asyncio.run(get_inputs())

            return cast(dict[str, torch.Tensor], inputs)

        trainer._prepare_inputs = _async_prepare_inputs

        return AsyncState(
            model=model,
            tokenizer=tokenizer,
            peft_model=peft_model,
            trainer=trainer,
            inputs_queue=inputs_queue,
            results_queue=results_queue,
        )

    async def start_openai_server(self, config: dev.OpenAIServerConfig | None) -> None:
        inference_gpu_ids = [1]
        logger.info("[ASYNC_SERVICE] Starting vLLM with weight update support")
        logger.info(f"[ASYNC_SERVICE]  inference_gpu_ids: {inference_gpu_ids}")

        art_root = Path(__file__).parent.parent  # Go up from unsloth/ to art/
        vllm_server_script = art_root / "vllm" / "in_flight_server.py"
        if not vllm_server_script.exists():
            raise FileNotFoundError(
                f"in_flight_server.py not found at {vllm_server_script}."
            )
        logger.info(
            f"[ASYNC_SERVICE]  using in_flight_server.py at: {vllm_server_script}"
        )
        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
            os.makedirs(os.path.dirname(lora_path), exist_ok=True)
            self.state.trainer.save_model(lora_path)
        config = dev.get_openai_server_config(
            model_name=self.model_name,
            base_model=self.base_model,
            log_file=f"{self.output_dir}/logs/vllm.log",
            # lora_path=lora_path,
            config=config,
        )
        # even if we are not using LoRA in vLLM, we need to enable so that the
        # tensor names will be the same
        engine_args = {**config.get("engine_args", {}), "enable_lora": True}
        logger.info(f"[ASYNC_SERVICE]  engine_args: {engine_args}")
        server_args = config.get("server_args", {})
        logger.info(f"[ASYNC_SERVICE]  server_args: {server_args}")
        vllm_args = [
            *[
                f"--{key.replace('_', '-')}{f'={item}' if item is not True else ''}"
                for args in [engine_args, server_args]
                for key, value in args.items()
                for item in (value if isinstance(value, list) else [value])
                if item is not None
            ],
        ]
        logger.info(f"[ASYNC_SERVICE]  vllm_args: {vllm_args}")
        inference_cmd = ["uv", "run", vllm_server_script, *vllm_args]
        log_dir = os.path.join(self.output_dir, "logs")
        logger.info(f"[ASYNC_SERVICE]  log_dir: {log_dir}")
        os.makedirs(log_dir, exist_ok=True)
        with open(Path(log_dir) / "inference.stdout", "w") as log_file:
            self._vllm_process = subprocess.Popen(
                inference_cmd,
                env={
                    **os.environ,
                    # "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, inference_gpu_ids)),
                },
                stdout=log_file,
                stderr=log_file,
            )
        logger.info(
            f"[PIPELINE_RL_SERVICE] vLLM process started (PID: {self._vllm_process.pid})"
        )
        # start loading unsloth
        _ = self.state
        server_config = config or dev.OpenAIServerConfig()
        server_args = server_config.get("server_args", {})
        base_url = f"http://{server_args.get('host', '0.0.0.0')}:{server_args.get('port', 8000)}/v1"
        await self._wait_for_vllm_ready(base_url, server_args.get("api_key", "default"))
        logger.info("[ASYNC_SERVICE] vLLM server is ready")

        # Initialize broadcaster
        await self._init_broadcaster(server_args)
        await self._update_weights_via_nccl()

    async def _wait_for_vllm_ready(self, base_url: str, api_key: str) -> None:
        """
        Wait for vLLM server to be ready by polling the /v1/models endpoint.
        """
        import time

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > 120:
                raise TimeoutError(
                    f"vLLM server at {base_url} did not become ready within 120s."
                )
            try:
                async for _ in client.models.list():
                    return
            except:  # noqa: E722
                await asyncio.sleep(0.1)

    async def _init_broadcaster(self, server_args: ServerArgs) -> None:
        """
        Initialize the broadcaster by calling the /init_broadcaster endpoint.
        """

        host = server_args.get("host", "0.0.0.0")
        port = server_args.get("port", 8000)
        server_url = f"http://{host}:{port}"

        payload = {
            "host": "0.0.0.0",
            "port": 29500,
            "server_rank": 0,
            "num_inference_server": 1,
            "timeout": 300,
        }

        async def _init_nccl_broadcast_sender() -> NCCLBroadcastSender:
            logger.info(
                f"[ASYNC_SERVICE] Initializing NCCLBroadcastSender on device {torch.cuda.current_device()}"
            )
            broadcast_sender = NCCLBroadcastSender(
                host="0.0.0.0",
                port=29500,
                rank=0,
                world_size=2,
                device=torch.cuda.current_device(),
                logger=logger,
                timeout=300,
            )
            logger.info(
                f"[ASYNC_SERVICE] NCCLBroadcastSender initialized on device {torch.cuda.current_device()}"
            )
            return broadcast_sender

        async def _init_nccl_broadcast_receiver() -> None:
            async with httpx.AsyncClient(timeout=300) as client:
                try:
                    logger.info(
                        f"[ASYNC_SERVICE] Initializing NCCLBroadcastReceiver with url: {server_url}/init_broadcaster"
                    )
                    response = await client.post(
                        f"{server_url}/init_broadcaster",
                        json=payload,
                    )
                    response.raise_for_status()
                    logger.info("[ASYNC_SERVICE] Broadcaster initialized successfully")
                except Exception as e:
                    logger.warning(
                        f"[ASYNC_SERVICE] Failed to initialize broadcaster: {e}"
                    )

        # TODO: Instead of sleeping, we can consider using different processes/background threads to ensure the sender does not block
        # 1. Initialize the receiver and wait for 1 second to ensure the POST request is sent
        receiver_task = asyncio.create_task(_init_nccl_broadcast_receiver())
        await asyncio.sleep(1)
        # 2. Initialize the sender (blocking)
        broadcast_sender = await _init_nccl_broadcast_sender()
        # 3. Wait for the receiver to finish initializing
        await receiver_task

        self.nccl_broadcast = broadcast_sender
        logger.info(
            f"[ASYNC_SERVICE] NCCLBroadcastSender initialized: {self.nccl_broadcast}"
        )

    async def vllm_engine_is_sleeping(self) -> bool:
        return False

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        # Get the packed tensors from disk
        packed_tensors = packed_tensors_from_dir(**disk_packed_tensors)
        # Wait for existing batches to finish
        await self.state.results_queue.join()
        # If we haven't already, start the training task
        if self._train_task is None:
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
        # Enter training mode
        for offset in range(0, packed_tensors["tokens"].shape[0]):
            for _ in range(2 if warmup else 1):
                if precalculate_logprobs and not warmup:
                    packed_tensors["original_logprobs"] = packed_tensors["logprobs"]  # type: ignore
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
                # Wait for a result from the queue or for the training task to,
                # presumably, raise an exception
                done, _ = await asyncio.wait(
                    [
                        asyncio.create_task(self.state.results_queue.get()),
                        self._train_task,
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if verbose:
                    print(
                        "Done waiting for a result from the queue or for the training task to, presumably, raise an exception"
                    )
                for task in done:
                    result = task.result()
                    # If `result` is `None`, the training task finished somehow.
                    assert result is not None, "The training task should never finish."
                    self.state.results_queue.task_done()
                    if warmup:
                        from .train import gc_and_empty_cuda_cache

                        gc_and_empty_cuda_cache()
                        await asyncio.sleep(0.1)
                        warmup = False
                    else:
                        yield result
        if verbose:
            print("Saving new LoRA adapter...")
        # Save the new LoRA adapter
        next_step = get_step_from_dir(self.output_dir) + 1
        checkpoint_dir = get_step_checkpoint_dir(self.output_dir, next_step)
        os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
        self.state.trainer.save_model(checkpoint_dir)
        if verbose:
            print("Setting new LoRA adapter...")
        # Set the new LoRA adapter
        await self._update_weights_via_nccl()
        if verbose:
            print("New LoRA adapter set")

        if verbose:
            print("ModelService.train complete")

    async def _update_weights_via_nccl(self) -> None:
        """Triggers the weight update on the vLLM engine via NCCL."""
        # Note: The /update_weights endpoint is on the root, not under /v1
        if not self.nccl_broadcast:
            logger.warning("NCCL broadcaster not initialized")
            raise RuntimeError

        logger.info("Merging adapter temporarily for broadcasting...")

        merged_wrapper = None
        try:
            # Merge adapters into base model weights
            self.state.peft_model.merge_adapter()

            # Extract state dict from base model
            # We need to access the underlying model which has the merged weights
            state_dict = self.state.peft_model.base_model.model.state_dict()

            # Filter out lora keys if any remain (usually merge_adapter merges them into the weights)
            # but we want to be sure we are sending clean weights
            logger.info(
                f"[ASYNC_SERVICE] DEBUG: State dict number of keys: {len(state_dict.keys())}"
            )
            clean_state_dict = {k: v for k, v in state_dict.items() if "lora_" not in k}
            logger.info(
                f"[ASYNC_SERVICE] DEBUG: Clean state dict number of keys: {len(clean_state_dict.keys())}"
            )
            merged_wrapper = StateDictWrapper(clean_state_dict)

            async def _broadcast():
                logger.info("Broadcasting state dict...")
                if merged_wrapper:
                    await asyncio.to_thread(
                        self.nccl_broadcast.broadcast_state_dict, merged_wrapper
                    )

            async def _trigger():
                # Wait a bit to ensure the broadcaster is ready/started before we tell the server to receive
                # Although asyncio.gather starts them "concurrently", the broadcast operation is heavy
                # and we want to make sure we don't timeout the HTTP request if the broadcast takes long to prep?
                # Actually, the server will block on receive_state_dict immediately upon receiving the request.
                # The sender (us) will block on broadcast_state_dict.
                # We just need both to happen.
                base_url = "http://localhost:8000"
                logger.info("Triggering inference server update...")
                # We might need a slight delay or just rely on the fact that broadcast_state_dict takes time to setup?
                # Ideally we just fire both.
                await asyncio.sleep(
                    0.1
                )  # Small yield to ensure broadcast thread starts
                async with httpx.AsyncClient(timeout=300) as client:
                    response = await client.post(
                        f"{base_url}/update_weights",
                    )
                    response.raise_for_status()
                    logger.info(
                        f"[ASYNC_SERVICE] DEBUG: State dict update triggered successfully"
                    )

            await asyncio.gather(_broadcast(), _trigger())

        finally:
            logger.info("Unmerging adapter to restore training state...")
            self.state.peft_model.unmerge_adapter()
            # Clean up memory
            del merged_wrapper

    async def _update_lora_weights_via_nccl(self) -> None:
        """Triggers the weight update on the vLLM engine via NCCL."""
        # Note: The /update_weights endpoint is on the root, not under /v1
        if not self.nccl_broadcast:
            logger.warning("NCCL broadcaster not initialized")
            raise RuntimeError

        # Extract PEFT config
        logger.info(
            f"[ASYNC_SERVICE] DEBUG: Peft config: {self.state.peft_model.peft_config}"
        )
        peft_config = self.state.peft_model.peft_config["default"]
        logger.info(f"[ASYNC_SERVICE] DEBUG: Peft config (default): {peft_config}")
        # Convert to dict if it's an object, or use it directly
        config_dict = (
            peft_config.to_dict() if hasattr(peft_config, "to_dict") else peft_config
        )
        logger.info(f"[ASYNC_SERVICE] DEBUG: Peft config (dict): {config_dict}")
        # Filter for relevant fields for PEFTHelper
        # r, lora_alpha, target_modules, bias, modules_to_save, use_rslora, use_dora
        relevant_keys = [
            "r",
            "lora_alpha",
            "target_modules",
            "bias",
            "modules_to_save",
            "use_rslora",
            "use_dora",
        ]
        filtered_config = {
            k: config_dict.get(k) for k in relevant_keys if k in config_dict
        }
        logger.info(f"[ASYNC_SERVICE] DEBUG: Filtered config: {filtered_config}")
        logger.info(
            f"[ASYNC_SERVICE] DEBUG: Broadcasting LoRA adapter weights keys: {self.state.peft_model.state_dict().keys()}"
        )

        async def _broadcast():
            await asyncio.to_thread(
                self.nccl_broadcast.broadcast_lora,
                self.state.peft_model,
                peft_config=filtered_config,
            )

        async def _trigger():
            logger.info("Triggering inference server update...")
            base_url = "http://localhost:8000"
            await asyncio.sleep(0.1)
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    f"{base_url}/update_lora_weights",
                )
                response.raise_for_status()
                logger.info(
                    f"[ASYNC_SERVICE] DEBUG: LoRA adapter update triggered successfully"
                )

        await asyncio.gather(_broadcast(), _trigger())

    async def _set_lora(self, lora_path: str) -> None:
        """Sets the LoRA adapter with ID 1 in the vLLM engine."""
        logger.info(f"[ASYNC_SERVICE] DEBUG: Setting LoRA adapter: {lora_path}")
        base_url = f"http://localhost:8000"
        api_key = "default"
        # unload the old LoRA adapter
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{base_url}/v1/unload_lora_adapter",
                json={"lora_name": self.model_name, "lora_int_id": 1},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            logger.info(f"[ASYNC_SERVICE] DEBUG: LoRA adapter unloaded successfully")
        # load the new LoRA adapter
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{base_url}/v1/load_lora_adapter",
                json={
                    "lora_name": self.model_name,
                    "lora_int_id": 1,
                    "lora_path": lora_path,
                },
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            logger.info(f"[ASYNC_SERVICE] DEBUG: LoRA adapter loaded successfully")

    def close(self) -> None:
        vllm_process = getattr(self, "_vllm_process", None)
        if vllm_process:
            logger.info("[ASYNC_SERVICE] Terminating vLLM process...")
            vllm_process.terminate()
            try:
                vllm_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "[ASYNC_SERVICE] vLLM process did not terminate, killing..."
                )
                vllm_process.kill()
                vllm_process.wait()
            logger.info("[ASYNC_SERVICE] vLLM process terminated")
