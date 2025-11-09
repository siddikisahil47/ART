import asyncio
from datetime import datetime
import json
import logging
import math
import os
import subprocess
from types import TracebackType
from typing import AsyncIterator, Callable, Literal, cast
import warnings

import aiohttp
import numpy as np
from openai import AsyncOpenAI
import polars as pl
import torch
from tqdm import auto as tqdm
from transformers import AutoImageProcessor, AutoTokenizer
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import Self
import wandb
from wandb.sdk.wandb_run import Run
import weave
from weave.trace.weave_client import WeaveClient

from art.utils.old_benchmarking.calculate_step_metrics import calculate_step_std_dev
from art.utils.output_dirs import (
    get_default_art_path,
    get_model_dir,
    get_output_dir_from_model_properties,
    get_step_checkpoint_dir,
    get_trajectories_split_dir,
)
from art.utils.s3 import (
    ExcludableOption,
    pull_model_from_s3,
    push_model_to_s3,
)
from art.utils.trajectory_logging import write_trajectory_groups_parquet
from mp_actors import close_proxy, move_to_child_process

from .. import dev
from ..backend import Backend
from ..model import Model, TrainableModel
from ..preprocessing.pack import (
    PackedTensors,
    packed_tensors_from_tokenized_results,
    packed_tensors_to_dir,
    plot_packed_tensors,
)
from ..preprocessing.tokenize import tokenize_trajectory_groups
from ..trajectories import Trajectory, TrajectoryGroup
from ..types import Message, TrainConfig
from ..utils import format_message, get_model_step
from .checkpoints import (
    delete_checkpoints,
)
from .service import ModelService

logger = logging.getLogger(__name__)


class LocalBackend(Backend):
    def __init__(self, *, in_process: bool = False, path: str | None = None) -> None:
        """
        Initializes a local, directory-based Backend interface at the given path.

        Note:
            The local Backend uses Weights & Biases for training monitoring.
            If you don't have a W&B account, you can create one at https://wandb.ai.

        Args:
            in_process: Whether to run the local service in-process.
            path: The path to the local directory. Defaults to "{repo_root}/.art".
        """
        self._in_process = in_process
        self._path = path or get_default_art_path()
        os.makedirs(self._path, exist_ok=True)

        # Other initialization
        self._services: dict[str, ModelService] = {}
        self._tokenizers: dict[str, PreTrainedTokenizerBase] = {}
        self._image_processors: dict[str, BaseImageProcessor | None] = {}
        self._wandb_runs: dict[str, Run] = {}
        self._weave_clients: dict[str, WeaveClient] = {}
        self._actor_update_group: torch.distributed.ProcessGroup | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._close()

    async def close(self) -> None:
        """
        If running vLLM in a separate process, this will kill that process and close the communication threads.
        """
        self._close()

    def _close(self) -> None:
        for _, service in self._services.items():
            close_proxy(service)

    async def register(
        self,
        model: Model,
    ) -> None:
        """
        Registers a model with the local Backend for logging and/or training.

        Args:
            model: An art.Model instance.
        """
        output_dir = get_model_dir(model=model, art_path=self._path)
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/model.json", "w") as f:
            json.dump(model.model_dump(), f)

        # Auto-migrate any old JSONL trajectory files to Parquet
        from art.utils.trajectory_migration import auto_migrate_on_register

        auto_migrate_on_register(output_dir)

        # Initialize wandb and weave early if this is a trainable model
        if model.trainable and "WANDB_API_KEY" in os.environ:
            _ = self._get_wandb_run(model)

    async def _get_service(self, model: TrainableModel) -> ModelService:
        from ..dev.get_model_config import get_model_config

        if model.name not in self._services:
            config = get_model_config(
                base_model=model.base_model,
                output_dir=get_model_dir(model=model, art_path=self._path),
                config=model._internal_config,
            )
            is_tinker = config.get("tinker_args") is not None
            if is_tinker:
                from ..tinker.service import TinkerService

                service_class = TinkerService
            elif config.get("torchtune_args") is not None:
                from ..torchtune.service import TorchtuneService

                service_class = TorchtuneService
            elif config.get("_use_pipeline_rl", False):
                from .pipeline_rl_service import PipelineRLService

                service_class = PipelineRLService
            else:
                from ..unsloth.service import UnslothService

                service_class = UnslothService
                # When moving the service to a child process, import unsloth
                # early to maximize optimizations
                os.environ["IMPORT_UNSLOTH"] = "1"
            self._services[model.name] = service_class(
                model_name=model.name,
                base_model=model.base_model,
                config=config,
                output_dir=get_model_dir(model=model, art_path=self._path),
            )
            if not self._in_process:
                # Kill all "model-service" processes to free up GPU memory
                subprocess.run(["pkill", "-9", "model-service"])
                self._services[model.name] = move_to_child_process(
                    self._services[model.name],
                    process_name="tinker-service" if is_tinker else "model-service",
                )
        return self._services[model.name]

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
        model: TrainableModel,
        num_actor_gpus: int,
    ) -> tuple[str, int]:
        """
        Initialize process group for weight updates (Phase 1: setup infrastructure).

        This creates the TCP rendezvous endpoint that will be used by both the trainer
        and vLLM to coordinate weight updates via NCCL.

        Args:
            model: The trainable model
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

        client = AsyncOpenAI(api_key="EMPTY", base_url=f"{base_url}/v1")
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

    def _get_packed_tensors(
        self,
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        advantage_balance: float,
        allow_training_without_logprobs: bool,
        scale_rewards: bool,
        plot_tensors: bool,
    ) -> PackedTensors | None:
        if model.base_model not in self._tokenizers:
            self._tokenizers[model.base_model] = AutoTokenizer.from_pretrained(
                model.base_model
            )
        if model.base_model not in self._image_processors:
            try:
                self._image_processors[model.base_model] = (
                    AutoImageProcessor.from_pretrained(model.base_model, use_fast=True)
                )
            except Exception:
                self._image_processors[model.base_model] = None
        tokenizer = self._tokenizers[model.base_model]
        tokenized_results = list(
            tokenize_trajectory_groups(
                tokenizer,
                trajectory_groups,
                allow_training_without_logprobs,
                scale_rewards,
                image_processor=self._image_processors[model.base_model],
            )
        )
        if not tokenized_results:
            logger.warning("No tokenized_results")
            logger.warning(f"example group: {trajectory_groups[0]}")
            return None
        max_tokens = max(len(result.tokens) for result in tokenized_results)
        # Round up max_tokens to the nearest multiple of 2048
        sequence_length = math.ceil(max_tokens / 2048) * 2048
        # Cap sequence length at the model's max sequence length
        sequence_length = min(
            sequence_length,
            (model._internal_config or dev.InternalModelConfig())
            .get("init_args", {})
            .get("max_seq_length", 32_768),
        )
        packed_tensors = packed_tensors_from_tokenized_results(
            tokenized_results,
            sequence_length,
            pad_token_id=tokenizer.eos_token_id,  # type: ignore
            advantage_balance=advantage_balance,
        )
        if (
            not allow_training_without_logprobs
            and np.isnan(packed_tensors["logprobs"]).all()
        ):
            logger.warning(
                "There are no assistant logprobs to train on. Did you forget to include at least one Choice in Trajectory.messages_and_choices?"
            )
            logger.warning(f"example group: {trajectory_groups[0]}")
            return None
        if plot_tensors:
            plot_packed_tensors(
                packed_tensors, get_model_dir(model=model, art_path=self._path)
            )
        else:
            logger.info(
                f"Packed {len(tokenized_results)} trajectories into {packed_tensors['tokens'].shape[0]} sequences of length {packed_tensors['tokens'].shape[1]}"
            )
        return packed_tensors

    async def _get_step(self, model: TrainableModel) -> int:
        return self.__get_step(model)

    def __get_step(self, model: Model) -> int:
        if model.trainable:
            model = cast(TrainableModel, model)
            return get_model_step(model, self._path)
        # Non-trainable models do not have checkpoints/steps; default to 0
        return 0

    async def _delete_checkpoints(
        self,
        model: TrainableModel,
        benchmark: str,
        benchmark_smoothing: float,
    ) -> None:
        from ..tinker.service import TinkerService

        output_dir = get_model_dir(model=model, art_path=self._path)
        # Keep the latest step
        steps_to_keep = [get_model_step(model, self._path)]
        try:
            best_step = (
                pl.read_ndjson(f"{output_dir}/history.jsonl")
                .drop_nulls(subset=[benchmark])
                .group_by("step")
                .mean()
                .with_columns(pl.col(benchmark).ewm_mean(alpha=benchmark_smoothing))
                .sort(benchmark)
                .select(pl.col("step").last())
                .item()
            )
            steps_to_keep.append(best_step)
        except FileNotFoundError:
            print(f'"{output_dir}/history.jsonl" not found')
        except pl.exceptions.ColumnNotFoundError:
            print(f'No "{benchmark}" metric found in history')
        service = await self._get_service(model)
        if isinstance(service, TinkerService):
            await service.delete_checkpoints(steps_to_keep)
        else:
            delete_checkpoints(output_dir, steps_to_keep)

    async def _prepare_backend_for_training(
        self,
        model: TrainableModel,
        config: dev.OpenAIServerConfig | None = None,
    ) -> tuple[str, str]:
        service = await self._get_service(model)
        await service.start_openai_server(config=config)
        server_args = (config or {}).get("server_args", {})

        base_url = f"http://{server_args.get('host', '0.0.0.0')}:{server_args.get('port', 8000)}/v1"
        api_key = server_args.get("api_key", None) or "default"

        def done_callback(_: asyncio.Task[None]) -> None:
            close_proxy(self._services.pop(model.name))

        asyncio.create_task(
            self._monitor_openai_server(model.name, base_url, api_key)
        ).add_done_callback(done_callback)

        return base_url, api_key

    async def _monitor_openai_server(
        self, model_name: str, base_url: str, api_key: str
    ) -> None:
        openai_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        consecutive_failures = 0
        max_consecutive_failures = 3
        async with aiohttp.ClientSession() as session:
            while True:
                # Wait 30 seconds before checking again
                await asyncio.sleep(30)
                try:
                    # If the server is sleeping, skip the check
                    if await self._services[model_name].vllm_engine_is_sleeping():
                        consecutive_failures = 0
                        continue
                    # Check the metrics with a timeout
                    async with session.get(
                        f"{base_url.split('/v1')[0]}/metrics",
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        metrics = await response.text()
                    # Parse Prometheus metrics for running requests
                    running_requests = 0
                    pending_requests = 0
                    for line in metrics.split("\n"):
                        if line.startswith("vllm:num_requests_running"):
                            running_requests = int(float(line.split()[1]))
                        elif line.startswith("vllm:num_requests_waiting"):
                            pending_requests = int(float(line.split()[1]))
                    # If there are no running or pending requests, send a health check
                    if running_requests == 0 and pending_requests == 0:
                        try:
                            # Send a health check with a short timeout
                            await openai_client.completions.create(
                                model=model_name,
                                prompt="Hi",
                                max_tokens=1,
                                timeout=float(
                                    os.environ.get("ART_SERVER_MONITOR_TIMEOUT", 5.0)
                                ),
                            )
                        except Exception as e:
                            # If the server is sleeping, a failed health check is okay
                            if await self._services[
                                model_name
                            ].vllm_engine_is_sleeping():
                                consecutive_failures = 0
                                continue
                            raise e
                    # Reset failure counter on success
                    consecutive_failures = 0
                except Exception:
                    # If the server is sleeping during an exception, it's okay
                    try:
                        if await self._services[model_name].vllm_engine_is_sleeping():
                            consecutive_failures = 0
                            continue
                    except Exception:
                        pass  # If we can't check sleeping status, count it as a failure
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        raise
                    # Otherwise, continue and try again

    async def _log(
        self,
        model: Model,
        trajectory_groups: list[TrajectoryGroup],
        split: str = "val",
    ) -> None:
        # Save logs for trajectory groups
        parent_dir = get_trajectories_split_dir(
            get_model_dir(model=model, art_path=self._path), split
        )
        os.makedirs(parent_dir, exist_ok=True)

        # Get the file name for the current iteration, or default to 0 for non-trainable models
        iteration = self.__get_step(model)
        file_name = f"{iteration:04d}.parquet"

        # Write the logs to Parquet file (with ZSTD compression)
        write_trajectory_groups_parquet(trajectory_groups, f"{parent_dir}/{file_name}")

        # Collect all metrics (including reward) across all trajectories
        all_metrics: dict[str, list[float]] = {"reward": [], "exception_rate": []}

        for group in trajectory_groups:
            for trajectory in group:
                if isinstance(trajectory, BaseException):
                    all_metrics["exception_rate"].append(1)
                    continue
                else:
                    all_metrics["exception_rate"].append(0)
                # Add reward metric
                all_metrics["reward"].append(trajectory.reward)

                # Collect other custom metrics
                for metric, value in trajectory.metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(float(value))

        # Calculate averages for all metrics
        averages = {}
        for metric, values in all_metrics.items():
            if len(values) > 0:
                averages[metric] = sum(values) / len(values)

        # Calculate average standard deviation of rewards within groups
        averages["reward_std_dev"] = calculate_step_std_dev(trajectory_groups)

        self._log_metrics(model, averages, split)

    def _trajectory_log(self, trajectory: Trajectory) -> str:
        """Format a trajectory into a readable log string."""
        header = f"reward: {trajectory.reward} {' '.join(f'{k}: {v}' for k, v in trajectory.metrics.items())}\n\n"
        formatted_messages = []
        for message_or_choice in trajectory.messages_and_choices:
            if isinstance(message_or_choice, dict):
                message = message_or_choice
            else:
                message = cast(Message, message_or_choice.message.model_dump())
            formatted_messages.append(format_message(message))
        return header + "\n".join(formatted_messages)

    async def _train_model(
        self,
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        if verbose:
            print("Starting _train_model")
        service = await self._get_service(model)
        if verbose:
            print("Logging training data to disk...")
        await self._log(model, trajectory_groups, "train")
        if verbose:
            print("Packing tensors...")

        # Count submitted groups and trainable groups
        num_groups_submitted = len(trajectory_groups)
        num_groups_trainable = sum(
            1
            for group in trajectory_groups
            if group and len(set(trajectory.reward for trajectory in group)) > 1
        )

        packed_tensors = self._get_packed_tensors(
            model,
            trajectory_groups,
            advantage_balance=dev_config.get("advantage_balance", 0.0),
            allow_training_without_logprobs=dev_config.get(
                "allow_training_without_logprobs", False
            ),
            scale_rewards=dev_config.get("scale_rewards", True),
            plot_tensors=dev_config.get("plot_tensors", False),
        )
        if packed_tensors is None:
            print(
                "Skipping tuning as there is no suitable data. "
                "This can happen when all the trajectories in the same group "
                "have the same reward and thus no advantage to train on."
            )

            # Still advance the step by renaming the checkpoint directory
            current_step = self.__get_step(model)
            next_step = current_step + 1
            current_checkpoint_dir = get_step_checkpoint_dir(
                get_model_dir(model=model, art_path=self._path), current_step
            )
            next_checkpoint_dir = get_step_checkpoint_dir(
                get_model_dir(model=model, art_path=self._path), next_step
            )

            # If the current checkpoint exists, rename it to the next step
            if os.path.exists(current_checkpoint_dir):
                os.rename(current_checkpoint_dir, next_checkpoint_dir)
                print(
                    f"Advanced step from {current_step} to {next_step} (no training occurred)"
                )

            # Log metrics showing no groups were trainable
            self._log_metrics(
                model,
                {
                    "num_groups_submitted": num_groups_submitted,
                    "num_groups_trainable": 0,
                },
                "train",
                step=next_step,
            )
            return
        disk_packed_tensors = packed_tensors_to_dir(
            packed_tensors, f"{get_model_dir(model=model, art_path=self._path)}/tensors"
        )
        if dev_config.get("scale_learning_rate_by_reward_std_dev", False):
            config = config.model_copy(
                update={
                    "learning_rate": config.learning_rate
                    * self._get_reward_std_dev_learning_rate_multiplier(model)
                }
            )
        results: list[dict[str, float]] = []
        estimated_gradient_steps = disk_packed_tensors["num_sequences"]
        if torchtune_args := (model._internal_config or dev.InternalModelConfig()).get(
            "torchtune_args"
        ):
            tp = torchtune_args.get("tensor_parallel_dim", 1)
            cp = torchtune_args.get("context_parallel_dim", 1)
            world_size = torch.cuda.device_count()
            dp = world_size // (tp * cp)
            estimated_gradient_steps = math.ceil(estimated_gradient_steps / dp)
        pbar = tqdm.tqdm(total=estimated_gradient_steps, desc="train")
        async for result in service.train(
            disk_packed_tensors, config, dev_config, verbose
        ):
            num_gradient_steps = int(
                result.pop("num_gradient_steps", estimated_gradient_steps)
            )
            assert num_gradient_steps == estimated_gradient_steps, (
                f"num_gradient_steps {num_gradient_steps} != estimated_gradient_steps {estimated_gradient_steps}"
            )
            results.append(result)
            yield {**result, "num_gradient_steps": num_gradient_steps}
            pbar.update(1)
            pbar.set_postfix(result)
        pbar.close()
        if verbose:
            print("Logging metrics...")
        data = {
            k: sum(d.get(k, 0) for d in results) / sum(1 for d in results if k in d)
            for k in {k for d in results for k in d}
        }
        # Add group counting metrics
        data["num_groups_submitted"] = num_groups_submitted
        data["num_groups_trainable"] = num_groups_trainable
        # Get the current step after training
        current_step = self.__get_step(model)
        self._log_metrics(model, data, "train", step=current_step)
        if verbose:
            print("_train_model complete")

    async def _generate_batch(
        self,
        model: TrainableModel,
        prompts: list[list[dict]],
        reward_fn: Callable,
        base_url: str = "http://localhost:8000",
        generation_config: dict | None = None,
        trajectories_per_prompt: int = 32,
    ) -> list[TrajectoryGroup]:
        """
        Generate trajectories by calling vLLM via OpenAI-compatible API.

        This function is reusable across both sequential and concurrent implementations.
        For each prompt, it generates N trajectories (multiple completions) which are
        grouped together for advantage calculation in RL training.

        Args:
            model: The TrainableModel being trained
            prompts: List of conversation prompts (each prompt is a list of message dicts)
            reward_fn: Function that takes (prompt, response) and returns a reward (float)
            base_url: Base URL for vLLM server (default: http://localhost:8000)
            generation_config: Optional generation config (max_tokens, temperature, etc.)
            trajectories_per_prompt: Number of completions to generate per prompt (default: 32)

        Returns:
            List of TrajectoryGroup objects, one per prompt, each containing N trajectories

        Example:
            ```python
            def compute_reward(prompt, response):
                # Task-specific reward logic
                if "maybe" in response.lower():
                    return 1.0
                elif "no" in response.lower():
                    return 0.75
                else:
                    return 0.5

            prompts = [
                [{"role": "user", "content": "Respond with yes, no, or maybe"}],
                [{"role": "user", "content": "Say something"}],
            ]

            # Generate 32 trajectories for each prompt
            trajectory_groups = await backend._generate_batch(
                model=model,
                prompts=prompts,
                reward_fn=compute_reward,
                base_url="http://localhost:8000",
                trajectories_per_prompt=32,
            )
            # Result: 2 TrajectoryGroups, each with 32 trajectories
            ```
        """
        from ..trajectories import Trajectory, TrajectoryGroup

        logger.info(
            f"[GENERATE_BATCH] Generating {len(prompts)} groups × {trajectories_per_prompt} trajectories = {len(prompts) * trajectories_per_prompt} total"
        )
        logger.info(f"[GENERATE_BATCH]   Base URL: {base_url}")

        # Set default generation config
        if generation_config is None:
            generation_config = {
                "max_tokens": 512,
                "temperature": 0.7,
            }

        logger.info(f"[GENERATE_BATCH]   Generation config: {generation_config}")

        # Create OpenAI client
        client = AsyncOpenAI(api_key="EMPTY", base_url=f"{base_url}/v1")

        # Build list of all vLLM calls: each prompt repeated N times
        # This allows us to generate N different completions for each prompt
        logger.info(
            f"[GENERATE_BATCH] Calling vLLM for {len(prompts)} prompts × {trajectories_per_prompt} = {len(prompts) * trajectories_per_prompt} completions..."
        )

        # Create all completion tasks (one per trajectory)
        completion_tasks = []
        for prompt_idx, prompt in enumerate(prompts):
            for traj_idx in range(trajectories_per_prompt):
                task = client.chat.completions.create(
                    model=model.base_model,  # vLLM uses base model name
                    messages=prompt,
                    **generation_config,
                )
                completion_tasks.append((prompt_idx, traj_idx, prompt, task))

        try:
            # Execute all completions in parallel
            results = await asyncio.gather(
                *[task for (_, _, _, task) in completion_tasks]
            )
            logger.info(f"[GENERATE_BATCH] Received {len(results)} responses from vLLM")
        except Exception as e:
            logger.error(f"[GENERATE_BATCH] Error calling vLLM: {e}")
            raise

        # Group trajectories by prompt_idx
        trajectories_by_prompt: dict[int, list[Trajectory]] = {
            i: [] for i in range(len(prompts))
        }

        for (prompt_idx, traj_idx, prompt, _), response in zip(
            completion_tasks, results
        ):
            # Extract response content
            response_content = response.choices[0].message.content or ""

            # Compute reward using provided function
            reward = reward_fn(prompt, response_content)

            # Create trajectory
            # messages_and_choices format: [user_msg1, assistant_msg1, user_msg2, assistant_msg2, ...]
            messages_and_choices = list(prompt) + [
                {"role": "assistant", "content": response_content}
            ]

            trajectory = Trajectory(
                messages_and_choices=messages_and_choices,
                reward=reward,
                metadata={
                    "prompt_idx": prompt_idx,
                    "traj_idx": traj_idx,
                    "response_tokens": (
                        response.usage.completion_tokens if response.usage else 0
                    ),
                },
            )

            trajectories_by_prompt[prompt_idx].append(trajectory)

        # Create TrajectoryGroups (one per prompt, each with N trajectories)
        trajectory_groups = []
        for prompt_idx in range(len(prompts)):
            trajectories = trajectories_by_prompt[prompt_idx]
            trajectory_group = TrajectoryGroup(trajectories)
            trajectory_groups.append(trajectory_group)

        logger.info(
            f"[GENERATE_BATCH] Created {len(trajectory_groups)} trajectory groups"
        )
        logger.info(
            f"[GENERATE_BATCH]   Each group contains {trajectories_per_prompt} trajectories"
        )
        return trajectory_groups

    async def _pipeline_rl_train(
        self,
        model: TrainableModel,
        trajectory_groups_list: list[list[TrajectoryGroup]] | None = None,
        config: TrainConfig | None = None,
        dev_config: dev.TrainConfig | None = None,
        inference_gpu_ids: list[int] | None = None,
        trainer_gpu_ids: list[int] | None = None,
        max_step_lag: int = 5,
        base_url: str = "http://localhost:8000",
        concurrent: bool = False,
        prompt_generator: Callable | None = None,
        reward_fn: Callable | None = None,
        num_iterations: int | None = None,
        num_groups: int = 20,
        trajectories_per_group: int = 2,
    ) -> AsyncIterator[dict[str, float]]:
        """
        PipelineRL training with optional concurrent mode.

        Phase 1 (Sequential): Uses trajectory_groups_list, processes sequentially
        Phase 2 (Concurrent): Uses prompt_generator + reward_fn, runs generation
                              and training concurrently with backpressure

        Sequential Mode:
        1. Start vLLM with process group support
        2. For each training iteration:
           a. Use pre-generated trajectory groups
           b. Train on trajectories (blocking)
           c. Save LoRA checkpoint
           d. Swap LoRA in vLLM (via HTTP API)
           e. Repeat

        Concurrent Mode:
        1. Start vLLM with process group support
        2. Launch two concurrent tasks:
           a. Generation task: continuously generates trajectories
           b. Training task: continuously trains on incoming trajectories
        3. Backpressure via max_step_lag prevents generation from getting too far ahead
        4. LoRA swapping happens after each training step

        Args:
            model: The trainable model
            trajectory_groups_list: List of trajectory groups (sequential mode only)
            config: Training configuration
            dev_config: Developer configuration
            inference_gpu_ids: List of GPU IDs for vLLM inference (default: [0])
            trainer_gpu_ids: List of GPU IDs for training (default: [1])
            max_step_lag: Max steps ahead generation can be (for backpressure)
            base_url: Base URL for vLLM server (default: http://localhost:8000)
            concurrent: Enable concurrent mode (default: False)
            prompt_generator: Generator function for prompts (concurrent mode only)
            reward_fn: Reward function (prompt, response) -> float (concurrent mode)
            num_iterations: Number of iterations for concurrent mode
            num_groups: Number of trajectory groups per batch (concurrent mode)
            trajectories_per_group: Number of trajectories per group (for advantage calc)

        Yields:
            Training metrics for each gradient step
        """
        from .pipeline_rl_service import PipelineRLService

        # Set default GPU assignments if not provided
        if inference_gpu_ids is None:
            inference_gpu_ids = [0]
        if trainer_gpu_ids is None:
            trainer_gpu_ids = [1]

        num_actor_gpus = len(inference_gpu_ids)

        mode = "Concurrent" if concurrent else "Sequential"
        logger.info(f"[PIPELINE_RL] Starting {mode} Pipeline")
        logger.info(f"[PIPELINE_RL]   inference_gpu_ids: {inference_gpu_ids}")
        logger.info(f"[PIPELINE_RL]   trainer_gpu_ids: {trainer_gpu_ids}")
        logger.info(f"[PIPELINE_RL]   max_step_lag: {max_step_lag}")
        logger.info(f"[PIPELINE_RL]   base_url: {base_url}")
        if concurrent:
            logger.info(f"[PIPELINE_RL]   num_iterations: {num_iterations}")
            logger.info(
                f"[PIPELINE_RL]   num_groups: {num_groups}, trajectories_per_group: {trajectories_per_group}"
            )
        else:
            logger.info(
                f"[PIPELINE_RL]   num_iterations: {len(trajectory_groups_list)}"
            )

        # Step 1: Initialize process group infrastructure
        logger.info("[PIPELINE_RL] Step 1: Initializing process group infrastructure")
        init_method, world_size = await self._init_process_group_for_weight_updates(
            model, num_actor_gpus
        )

        # Step 2: Get or create PipelineRLService
        # We need to set CUDA_VISIBLE_DEVICES *before* creating the service
        # because @mp_actors.move creates a subprocess that inherits the environment.
        # Setting it inside the subprocess is too late (CUDA already initialized).
        logger.info("[PIPELINE_RL] Step 2: Getting PipelineRLService")
        logger.info(
            "[PIPELINE_RL]   Setting CUDA_VISIBLE_DEVICES for training subprocess..."
        )

        import os

        # Save original environment
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

        # Set CUDA_VISIBLE_DEVICES to only show trainer GPUs
        # This will be inherited by the subprocess created by @mp_actors.move
        trainer_gpu_str = ",".join(str(gpu_id) for gpu_id in trainer_gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = trainer_gpu_str
        logger.info(
            f"[PIPELINE_RL]   Temporarily set CUDA_VISIBLE_DEVICES={trainer_gpu_str}"
        )
        logger.info(f"[PIPELINE_RL]   (original was: {original_cuda_visible})")

        try:
            # Create the service - subprocess inherits the modified environment
            service = await self._get_service(model)
            logger.info(
                "[PIPELINE_RL]   PipelineRLService subprocess created successfully"
            )
        finally:
            # Restore original environment for the main process
            if original_cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            logger.info(
                f"[PIPELINE_RL]   Restored CUDA_VISIBLE_DEVICES to: {original_cuda_visible}"
            )

        # Step 3: Start vLLM with weight update support
        logger.info("[PIPELINE_RL] Step 3: Starting vLLM with weight update support")
        openai_config = dev_config.get("openai_server_config", None)
        await service.start_openai_server_with_weight_updates(
            config=openai_config,
            init_method=init_method,
            world_size=world_size,
            actor_idx=0,
            inference_gpu_ids=inference_gpu_ids,
        )

        # Step 4: Join process group as trainer (MUST be before waiting for vLLM!)
        # vLLM is blocking during startup waiting for trainer to join
        logger.info("[PIPELINE_RL] Step 4: Joining process group as trainer")
        logger.info("[PIPELINE_RL]   This will unblock vLLM startup...")
        self._actor_update_group = await self._join_process_group_as_trainer(
            init_method, world_size
        )

        # Step 5: Wait for vLLM HTTP server to be ready
        logger.info("[PIPELINE_RL] Step 5: Waiting for vLLM HTTP server to be ready")
        await self._wait_for_vllm_ready(base_url, timeout=120.0)
        logger.info("[PIPELINE_RL] vLLM is ready!")

        # Step 6: Choose execution mode
        if not concurrent:
            # Sequential mode (Phase 1)
            logger.info("[PIPELINE_RL] Step 6: Starting sequential training loop")
            async for metrics in self._pipeline_rl_train_sequential(
                model=model,
                service=service,
                trajectory_groups_list=trajectory_groups_list,
                config=config,
                dev_config=dev_config,
                max_step_lag=max_step_lag,
                base_url=base_url,
            ):
                yield metrics
        else:
            # Concurrent mode (Phase 2)
            logger.info("[PIPELINE_RL] Step 6: Starting concurrent training pipeline")
            async for metrics in self._pipeline_rl_train_concurrent(
                model=model,
                service=service,
                prompt_generator=prompt_generator,
                reward_fn=reward_fn,
                config=config,
                dev_config=dev_config,
                num_iterations=num_iterations,
                num_groups=num_groups,
                trajectories_per_group=trajectories_per_group,
                max_step_lag=max_step_lag,
                base_url=base_url,
            ):
                yield metrics

        logger.info("[PIPELINE_RL] Training complete!")

    async def _pipeline_rl_train_sequential(
        self,
        model: TrainableModel,
        service,
        trajectory_groups_list: list[list[TrajectoryGroup]],
        config: TrainConfig,
        dev_config: dev.TrainConfig,
        max_step_lag: int,
        base_url: str,
    ) -> AsyncIterator[dict[str, float]]:
        """Sequential training loop (extracted from original implementation)."""
        for iteration, trajectory_groups in enumerate(trajectory_groups_list):
            logger.info(f"[SEQUENTIAL] === Iteration {iteration} ===")
            logger.info(f"[SEQUENTIAL]   Generation step: {service._generation_step}")
            logger.info(f"[SEQUENTIAL]   Training step: {service._training_step}")
            logger.info(
                f"[SEQUENTIAL]   Lag: {service._generation_step - service._training_step}"
            )

            # Log trajectories
            logger.info(
                f"[SEQUENTIAL] Logging {len(trajectory_groups)} trajectory groups..."
            )
            await self._log(model, trajectory_groups, "train")

            # Pack tensors (reuse existing ART functionality)
            logger.info(
                f"[SEQUENTIAL] Packing {len(trajectory_groups)} trajectory groups..."
            )
            packed_tensors = self._get_packed_tensors(
                model,
                trajectory_groups,
                advantage_balance=dev_config.get("advantage_balance", 0.0),
                allow_training_without_logprobs=dev_config.get(
                    "allow_training_without_logprobs", False
                ),
                scale_rewards=dev_config.get("scale_rewards", True),
                plot_tensors=dev_config.get("plot_tensors", False),
            )

            if packed_tensors is None:
                logger.warning("[SEQUENTIAL] No suitable training data, skipping...")
                continue

            disk_packed_tensors = packed_tensors_to_dir(
                packed_tensors,
                f"{get_model_dir(model=model, art_path=self._path)}/tensors",
            )

            # Train on batch
            logger.info(f"[SEQUENTIAL] Training on batch...")
            async for metrics in service.train(disk_packed_tensors, config, dev_config):
                logger.info(f"[SEQUENTIAL]   Training metrics: {metrics}")
                yield metrics

            # Get checkpoint directory (service.train() already saved it)
            next_step = self.__get_step(model)
            checkpoint_dir = get_step_checkpoint_dir(
                get_model_dir(model=model, art_path=self._path), next_step
            )
            logger.info(f"[SEQUENTIAL] Checkpoint saved at: {checkpoint_dir}")

            # Swap LoRA in vLLM (Phase 1 approach: use HTTP API)
            logger.info(f"[SEQUENTIAL] Swapping LoRA checkpoint...")
            await service.swap_lora_checkpoint(checkpoint_dir, base_url=base_url)

            service._generation_step += 1
            service._training_step += 1

            # Check lag
            lag = service._generation_step - service._training_step
            logger.info(
                f"[SEQUENTIAL] Current lag: {lag} (max allowed: {max_step_lag})"
            )

            if lag >= max_step_lag:
                logger.warning(
                    f"[SEQUENTIAL] Generation is {lag} steps ahead, "
                    "would block in concurrent mode"
                )

    async def _pipeline_rl_train_concurrent(
        self,
        model: TrainableModel,
        service,
        prompt_generator: Callable,
        reward_fn: Callable,
        config: TrainConfig,
        dev_config: dev.TrainConfig,
        num_iterations: int,
        num_groups: int,
        trajectories_per_group: int,
        max_step_lag: int,
        base_url: str,
    ) -> AsyncIterator[dict[str, float]]:
        """
        Concurrent training pipeline (Phase 2).

        Runs two concurrent tasks:
        1. Generation task: continuously generates trajectories
        2. Training task: continuously trains on incoming trajectories

        Backpressure via max_step_lag prevents generation from getting too far ahead.
        """
        from ..trajectories import Trajectory, TrajectoryGroup

        logger.info("[CONCURRENT] Starting concurrent pipeline")
        logger.info(f"[CONCURRENT]   num_iterations: {num_iterations}")
        logger.info(f"[CONCURRENT]   num_groups: {num_groups}")
        logger.info(f"[CONCURRENT]   trajectories_per_group: {trajectories_per_group}")
        logger.info(
            f"[CONCURRENT]   max_step_lag: {max_step_lag} (not used yet, TODO: implement proper policy lag tracking)"
        )

        # Create queue for trajectory flow
        # Use reasonable queue size for backpressure (queue blocks when full)
        queue_maxsize = 50
        trajectory_queue: asyncio.Queue[list[TrajectoryGroup]] = asyncio.Queue(
            maxsize=queue_maxsize
        )
        logger.info(f"[CONCURRENT]   trajectory_queue maxsize: {queue_maxsize}")

        # Shared state for metrics yielding
        metrics_queue: asyncio.Queue[dict[str, float]] = asyncio.Queue()

        # Generation task (runs concurrently with training)
        async def generation_task():
            """Continuously generates trajectories and puts them in the queue."""
            try:
                for iteration in range(num_iterations):
                    logger.info(
                        f"[GENERATION] Starting generation step {service._generation_step}"
                    )

                    # Generate prompts for this step
                    logger.info(f"[GENERATION]   Generating {num_groups} prompts...")
                    prompts = []
                    for _ in range(num_groups):
                        prompt = prompt_generator()
                        prompts.append(prompt)
                    logger.info(f"[GENERATION]   Generated {len(prompts)} prompts")

                    # Call vLLM to generate trajectories
                    # For each prompt: generate trajectories_per_group completions
                    logger.info(
                        f"[GENERATION]   Calling vLLM to generate {num_groups} groups × {trajectories_per_group} trajectories..."
                    )
                    trajectory_groups = await self._generate_batch(
                        model=model,
                        prompts=prompts,
                        reward_fn=reward_fn,
                        base_url=base_url,
                        trajectories_per_prompt=trajectories_per_group,
                    )
                    logger.info(
                        f"[GENERATION]   Received {len(trajectory_groups)} trajectory groups from vLLM"
                    )

                    # Put in queue (will block if queue is full - natural backpressure)
                    logger.info(
                        f"[GENERATION]   Putting trajectories in queue (queue size: {trajectory_queue.qsize()}/{queue_maxsize})"
                    )
                    await trajectory_queue.put(trajectory_groups)

                    service._generation_step += 1
                    logger.info(
                        f"[GENERATION] Completed generation step {service._generation_step - 1}"
                    )
                    lag = service._training_step - service._generation_step
                    logger.info(
                        f"[GENERATION]   Step delta: {lag} (training - generation)"
                    )

                logger.info("[GENERATION] All generations complete, signaling end")
                # Signal end by putting None
                await trajectory_queue.put(None)

            except Exception as e:
                logger.error(f"[GENERATION] Error in generation task: {e}")
                import traceback

                traceback.print_exc()
                raise

        # Training task (runs concurrently with generation)
        async def training_task():
            """Continuously trains on trajectories from the queue."""
            try:
                while True:
                    # Get trajectories from queue
                    logger.info(
                        f"[TRAINING] Waiting for trajectories (queue size: {trajectory_queue.qsize()}/{queue_maxsize})"
                    )
                    trajectory_groups = await trajectory_queue.get()

                    # Check for end signal
                    if trajectory_groups is None:
                        logger.info("[TRAINING] Received end signal, stopping")
                        trajectory_queue.task_done()
                        break

                    logger.info(
                        f"[TRAINING] Starting training step {service._training_step}"
                    )
                    logger.info(
                        f"[TRAINING]   Received {len(trajectory_groups)} trajectory groups"
                    )

                    # Log trajectories
                    await self._log(model, trajectory_groups, "train")

                    # Pack tensors (same as sequential mode)
                    logger.info(f"[TRAINING]   Packing trajectories...")
                    packed_tensors = self._get_packed_tensors(
                        model,
                        trajectory_groups,
                        advantage_balance=dev_config.get("advantage_balance", 0.0),
                        allow_training_without_logprobs=dev_config.get(
                            "allow_training_without_logprobs", False
                        ),
                        scale_rewards=dev_config.get("scale_rewards", True),
                        plot_tensors=dev_config.get("plot_tensors", False),
                    )

                    if packed_tensors is None:
                        logger.warning(
                            "[TRAINING] No suitable training data, skipping..."
                        )
                        # Still increment training step
                        service._training_step += 1
                        trajectory_queue.task_done()

                        lag = service._training_step - service._generation_step
                        logger.info(
                            f"[TRAINING] Skipped training step {service._training_step - 1}"
                        )
                        logger.info(
                            f"[TRAINING]   Step delta: {lag} (training - generation)"
                        )
                        continue

                    disk_packed_tensors = packed_tensors_to_dir(
                        packed_tensors,
                        f"{get_model_dir(model=model, art_path=self._path)}/tensors",
                    )

                    # Train on batch
                    logger.info(f"[TRAINING]   Training...")
                    async for metrics in service.train(
                        disk_packed_tensors, config, dev_config
                    ):
                        logger.info(f"[TRAINING]     Metrics: {metrics}")
                        # Put metrics in queue to be yielded by main coroutine
                        await metrics_queue.put(metrics)

                    # Get checkpoint directory (service.train() already saved it)
                    next_step = self.__get_step(model)
                    checkpoint_dir = get_step_checkpoint_dir(
                        get_model_dir(model=model, art_path=self._path), next_step
                    )

                    # Swap LoRA (Phase 2.2 uses swap, Step 2.4 will use NCCL)
                    logger.info(f"[TRAINING]   Swapping LoRA checkpoint...")
                    await service.swap_lora_checkpoint(
                        checkpoint_dir, base_url=base_url
                    )

                    service._training_step += 1
                    trajectory_queue.task_done()

                    lag = service._training_step - service._generation_step
                    logger.info(
                        f"[TRAINING] Completed training step {service._training_step - 1}"
                    )
                    logger.info(
                        f"[TRAINING]   Step delta: {lag} (training - generation)"
                    )

                logger.info("[TRAINING] All training complete")
                # Signal end to metrics yielder
                await metrics_queue.put(None)

            except Exception as e:
                logger.error(f"[TRAINING] Error in training task: {e}")
                import traceback

                traceback.print_exc()
                raise

        # Start both tasks concurrently
        logger.info("[CONCURRENT] Launching generation and training tasks")
        gen_task = asyncio.create_task(generation_task())
        train_task = asyncio.create_task(training_task())

        try:
            # Yield metrics as they become available
            while True:
                metrics = await metrics_queue.get()
                if metrics is None:
                    break
                yield metrics

            # Wait for both tasks to complete
            logger.info("[CONCURRENT] Waiting for tasks to complete...")
            await asyncio.gather(gen_task, train_task)
            logger.info("[CONCURRENT] Both tasks completed successfully")

        except Exception as e:
            logger.error(f"[CONCURRENT] Error in concurrent pipeline: {e}")
            # Cancel both tasks on error
            gen_task.cancel()
            train_task.cancel()
            try:
                await asyncio.gather(gen_task, train_task, return_exceptions=True)
            except:
                pass
            raise

    def _get_reward_std_dev_learning_rate_multiplier(
        self, model: TrainableModel
    ) -> float:
        output_dir = get_model_dir(model=model, art_path=self._path)
        learning_rate_multiplier = 1.0  # Default prior
        try:
            std_dev_history = (
                pl.read_ndjson(f"{output_dir}/history.jsonl")
                .drop_nulls(subset=["train/reward_std_dev"])
                .group_by("step")
                .mean()
                .sort("step")
            )

            # Fit linear regression to std_dev_history
            if len(std_dev_history) > 1:
                steps = std_dev_history["step"].to_numpy()
                std_devs = std_dev_history["train/reward_std_dev"].to_numpy()

                # Fit linear regression: y = mx + b
                # polyfit returns [coefficient, intercept] for degree 1
                coefficient, intercept = np.polyfit(steps, std_devs, deg=1)

                # Get prediction for the last step
                last_step = steps[-1]
                last_step_prediction = coefficient * last_step + intercept
                last_step_actual = std_devs[-1]

                # Calculate R-squared and adjusted R-squared
                predictions = coefficient * steps + intercept
                ss_residual = np.sum((std_devs - predictions) ** 2)
                ss_total = np.sum((std_devs - np.mean(std_devs)) ** 2)
                r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

                # Adjusted R-squared accounts for sample size
                # For simple linear regression: adj_R² = 1 - (1 - R²) * (n - 1) / (n - 2)
                n_samples = len(steps)
                if n_samples > 2:
                    adjusted_r_squared = 1 - (1 - r_squared) * (n_samples - 1) / (
                        n_samples - 2
                    )
                else:
                    adjusted_r_squared = (
                        0  # Not enough samples for meaningful adjustment
                    )

                # Calculate learning rate multiplier
                # raw_multiplier = last_step_prediction / intercept (if intercept > 0)
                # adjusted by goodness of fit: multiplier = 1 + adj_R² * (raw_multiplier - 1)
                if intercept > 0:
                    raw_multiplier = last_step_prediction / intercept
                    # learning_rate_multiplier = 1 + adjusted_r_squared * (
                    #     raw_multiplier - 1
                    # )
                    learning_rate_multiplier = raw_multiplier
                else:
                    # If intercept <= 0, can't calculate meaningful ratio, stick with prior
                    raw_multiplier = 1.0
                    learning_rate_multiplier = 1.0

                print(f"Regression fitted: y = {coefficient:.6f}x + {intercept:.6f}")
                print(f"  Coefficient (slope): {coefficient:.6f}")
                print(f"  Intercept: {intercept:.6f}")
                print(f"  R-squared: {r_squared:.4f}")
                print(
                    f"  Adjusted R-squared: {adjusted_r_squared:.4f} (n={n_samples} samples)"
                )
                print(
                    f"  Last step ({last_step}) prediction: {last_step_prediction:.6f}"
                )
                print(f"  Last step actual value: {last_step_actual:.6f}")
                print(
                    f"  Prediction error: {abs(last_step_actual - last_step_prediction):.6f}"
                )
                print(f"  Raw LR multiplier (pred/intercept): {raw_multiplier:.4f}")
                print(f"  Adjusted LR multiplier: {learning_rate_multiplier:.4f}")
            else:
                print(
                    f"Not enough data points to fit regression (need at least 2, got {len(std_dev_history)})"
                )

        except FileNotFoundError:
            print(f'"{output_dir}/history.jsonl" not found')
        except pl.exceptions.ColumnNotFoundError:
            print(f'No "train/reward_std_dev" metric found in history')

        return learning_rate_multiplier

    def _log_metrics(
        self,
        model: Model,
        metrics: dict[str, float],
        split: str,
        step: int | None = None,
    ) -> None:
        metrics = {f"{split}/{metric}": value for metric, value in metrics.items()}
        step = step if step is not None else self.__get_step(model)

        with open(
            f"{get_model_dir(model=model, art_path=self._path)}/history.jsonl", "a"
        ) as f:
            f.write(
                json.dumps(
                    {
                        k: v for k, v in metrics.items() if v == v
                    }  # Filter out NaN values
                    | {"step": step, "recorded_at": datetime.now().isoformat()}
                )
                + "\n"
            )

        # If we have a W&B run, log the data there
        if run := self._get_wandb_run(model):
            # Mark the step metric itself as hidden so W&B doesn't create an automatic chart for it
            wandb.define_metric("training_step", hidden=True)

            # Enabling the following line will cause W&B to use the training_step metric as the x-axis for all metrics
            # wandb.define_metric(f"{split}/*", step_metric="training_step")
            run.log({"training_step": step, **metrics}, step=step)

    def _get_wandb_run(self, model: Model) -> Run | None:
        if "WANDB_API_KEY" not in os.environ:
            return None
        if (
            model.name not in self._wandb_runs
            or self._wandb_runs[model.name]._is_finished
        ):
            run = wandb.init(
                project=model.project,
                name=model.name,
                id=model.name,
                resume="allow",
                settings=wandb.Settings(
                    x_stats_open_metrics_endpoints={
                        "vllm": "http://localhost:8000/metrics",
                    },
                    x_stats_open_metrics_filters=(
                        "vllm.vllm:num_requests_waiting",
                        "vllm.vllm:num_requests_running",
                    ),
                ),
            )
            self._wandb_runs[model.name] = run
            os.environ["WEAVE_PRINT_CALL_LINK"] = os.getenv(
                "WEAVE_PRINT_CALL_LINK", "False"
            )
            os.environ["WEAVE_LOG_LEVEL"] = os.getenv("WEAVE_LOG_LEVEL", "CRITICAL")
            self._weave_clients[model.name] = weave.init(model.project)
        return self._wandb_runs[model.name]

    # ------------------------------------------------------------------
    # Experimental support for S3
    # ------------------------------------------------------------------

    async def _experimental_pull_model_checkpoint(
        self,
        model: "TrainableModel",
        *,
        step: int | Literal["latest"] | None = None,
        local_path: str | None = None,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
    ) -> str:
        """Pull a model checkpoint to a local path.

        For LocalBackend, this:
        1. When step is "latest" or None, checks both local storage and S3 (if provided)
           to find the latest checkpoint, preferring local if steps are equal
        2. If checkpoint exists locally, uses it (optionally copying to local_path)
        3. If checkpoint doesn't exist locally but s3_bucket is provided, pulls from S3
        4. Returns the final checkpoint path

        Args:
            model: The model to pull checkpoint for.
            step: The step to pull. Can be an int for a specific step,
                 or "latest" to pull the latest checkpoint. If None, pulls latest.
            local_path: Custom directory to save/copy the checkpoint to.
                       If None, returns checkpoint from backend's default art path.
            s3_bucket: S3 bucket to check/pull from. When step is "latest", both
                       local storage and S3 are checked to find the true latest.
            prefix: S3 prefix.
            verbose: Whether to print verbose output.

        Returns:
            Path to the local checkpoint directory.
        """
        # Determine which step to use
        resolved_step: int
        if step is None or step == "latest":
            # Check both local storage and S3 (if provided) for the latest checkpoint
            local_latest_step: int | None = None
            s3_latest_step: int | None = None

            # Get latest from local storage
            try:
                local_latest_step = get_model_step(model, self._path)
                if local_latest_step == 0:
                    # get_model_step returns 0 if no checkpoints exist
                    local_latest_step = None
            except Exception:
                local_latest_step = None

            # Get latest from S3 if bucket provided
            if s3_bucket is not None:
                from art.utils.s3_checkpoint_utils import (
                    get_latest_checkpoint_step_from_s3,
                )

                s3_latest_step = await get_latest_checkpoint_step_from_s3(
                    model_name=model.name,
                    project=model.project,
                    s3_bucket=s3_bucket,
                    prefix=prefix,
                )

            # Determine which source has the latest checkpoint
            if local_latest_step is None and s3_latest_step is None:
                raise ValueError(
                    f"No checkpoints found for {model.project}/{model.name} in local storage or S3"
                )
            elif local_latest_step is None:
                resolved_step = s3_latest_step  # type: ignore[assignment]
                if verbose:
                    print(f"Using latest checkpoint from S3: step {resolved_step}")
            elif s3_latest_step is None:
                resolved_step = local_latest_step
                if verbose:
                    print(
                        f"Using latest checkpoint from local storage: step {resolved_step}"
                    )
            elif local_latest_step >= s3_latest_step:
                # Prefer local if equal or greater
                resolved_step = local_latest_step
                if verbose:
                    print(
                        f"Using latest checkpoint from local storage: step {resolved_step} "
                    )
            else:
                resolved_step = s3_latest_step
                if verbose:
                    print(f"Using latest checkpoint from S3: step {resolved_step} ")
        else:
            resolved_step = step

        # Check if checkpoint exists in the original training location
        original_checkpoint_dir = get_step_checkpoint_dir(
            get_model_dir(model=model, art_path=self._path), resolved_step
        )

        # Step 1: Ensure checkpoint exists at original_checkpoint_dir
        if not os.path.exists(original_checkpoint_dir):
            if s3_bucket is None:
                raise FileNotFoundError(
                    f"Checkpoint not found at {original_checkpoint_dir} and no S3 bucket specified"
                )
            if verbose:
                print(f"Pulling checkpoint step {resolved_step} from S3...")
            await pull_model_from_s3(
                model_name=model.name,
                project=model.project,
                step=resolved_step,
                s3_bucket=s3_bucket,
                prefix=prefix,
                verbose=verbose,
                art_path=self._path,
                exclude=["logs", "trajectories"],
            )
            # Validate that the checkpoint was actually downloaded
            if not os.path.exists(original_checkpoint_dir) or not os.listdir(
                original_checkpoint_dir
            ):
                raise FileNotFoundError(f"Checkpoint step {resolved_step} not found")

        # Step 2: Handle local_path if provided
        if local_path is not None:
            if verbose:
                print(
                    f"Copying checkpoint from {original_checkpoint_dir} to {local_path}..."
                )
            import shutil

            os.makedirs(local_path, exist_ok=True)
            shutil.copytree(original_checkpoint_dir, local_path, dirs_exist_ok=True)
            if verbose:
                print(f"✓ Checkpoint copied successfully")
            return local_path

        if verbose:
            print(
                f"Checkpoint step {resolved_step} exists at {original_checkpoint_dir}"
            )
        return original_checkpoint_dir

    async def _experimental_pull_from_s3(
        self,
        model: Model,
        *,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        delete: bool = False,
        only_step: int | Literal["latest"] | None = None,
        # LocalBackend extensions (not part of the base interface)
        step: int | None = None,
        exclude: list[ExcludableOption] | None = None,
        latest_only: bool = False,
    ) -> None:
        """Download the model directory from S3 into local Backend storage. Right now this can be used to pull trajectory logs for processing or model checkpoints.

        .. deprecated::
            This method is deprecated. Use `_experimental_pull_model_checkpoint` instead.

        Args:
            model: The model to pull from S3.
            step: DEPRECATED. Use only_step instead.
            s3_bucket: The S3 bucket to pull from. If None, the default bucket will be used.
            prefix: The prefix to pull from S3. If None, the model name will be used.
            verbose: Whether to print verbose output.
            delete: Whether to delete the local model directory.
            exclude: List of directories to exclude from sync. Valid options: "checkpoints", "logs", "trajectories".
            latest_only: DEPRECATED. Use only_step="latest" instead.
            only_step: If specified, only pull this specific step. Can be an int for a specific step,
                      or "latest" to pull only the latest checkpoint. If None, pulls all steps.
        """
        warnings.warn(
            "_experimental_pull_from_s3 is deprecated. Use _experimental_pull_model_checkpoint instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Handle backward compatibility and new only_step parameter
        if only_step is None and latest_only:
            only_step = "latest"

        # Handle the only_step parameter
        if only_step is not None and step is None:
            if only_step == "latest":
                from art.utils.s3_checkpoint_utils import (
                    get_latest_checkpoint_step_from_s3,
                )

                latest_step = await get_latest_checkpoint_step_from_s3(
                    model_name=model.name,
                    project=model.project,
                    s3_bucket=s3_bucket,
                    prefix=prefix,
                )

                if latest_step is not None:
                    step = latest_step
                    if verbose:
                        print(f"Found latest checkpoint at step {step}")
                else:
                    if verbose:
                        print("No checkpoints found in S3")
                    return
            else:
                # only_step is an int
                step = only_step
                if verbose:
                    print(f"Pulling specific checkpoint at step {step}")

        await pull_model_from_s3(
            model_name=model.name,
            project=model.project,
            step=step,
            s3_bucket=s3_bucket,
            prefix=prefix,
            verbose=verbose,
            delete=delete,
            art_path=self._path,
            exclude=exclude,
        )

    async def _experimental_push_to_s3(
        self,
        model: Model,
        *,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        delete: bool = False,
    ) -> None:
        """Upload the model directory from local storage to S3."""
        await push_model_to_s3(
            model_name=model.name,
            project=model.project,
            s3_bucket=s3_bucket,
            prefix=prefix,
            verbose=verbose,
            delete=delete,
            art_path=self._path,
        )

    async def _experimental_fork_checkpoint(
        self,
        model: Model,
        from_model: str,
        from_project: str | None = None,
        from_s3_bucket: str | None = None,
        not_after_step: int | None = None,
        verbose: bool = False,
        prefix: str | None = None,
    ) -> None:
        """Fork a checkpoint from another model to initialize this model.

        Args:
            model: The model to fork to.
            from_model: The name of the model to fork from.
            from_project: The project of the model to fork from. Defaults to model.project.
            from_s3_bucket: Optional S3 bucket to pull the checkpoint from. If provided,
                will pull from S3 first. Otherwise, will fork from local disk.
            not_after_step: Optional step number. If provided, will copy the last saved
                checkpoint that is <= this step. Otherwise, copies the latest checkpoint.
            verbose: Whether to print verbose output.
            prefix: Optional S3 prefix for the bucket.
        """
        # Default from_project to model.project if not provided
        from_project = from_project or model.project

        # Get source and destination directories
        source_model_dir = get_output_dir_from_model_properties(
            project=from_project,
            name=from_model,
            art_path=self._path,
        )
        dest_model_dir = get_output_dir_from_model_properties(
            project=model.project,
            name=model.name,
            art_path=self._path,
        )

        # If S3 bucket is provided, pull from S3 first
        if from_s3_bucket is not None:
            if verbose:
                print(
                    f"DEBUG: Fork checkpoint - from_s3_bucket={from_s3_bucket}, not_after_step={not_after_step}"
                )

            # Determine which checkpoint to pull
            if not_after_step is None:
                # Pull only the latest checkpoint
                if verbose:
                    print(
                        f"Pulling latest checkpoint for model {from_model} from S3 bucket {from_s3_bucket}..."
                    )
                await self._experimental_pull_from_s3(
                    Model(name=from_model, project=from_project),
                    s3_bucket=from_s3_bucket,
                    verbose=verbose,
                    exclude=["logs", "trajectories"],
                    only_step="latest",
                )
            else:
                # Find the right checkpoint not after the specified step
                from art.utils.s3_checkpoint_utils import (
                    get_checkpoint_step_not_after_from_s3,
                )

                if verbose:
                    print(
                        f"Finding checkpoint not after step {not_after_step} for model {from_model} in S3..."
                    )

                # Find which step to pull
                target_step = await get_checkpoint_step_not_after_from_s3(
                    model_name=from_model,
                    project=from_project,
                    not_after_step=not_after_step,
                    s3_bucket=from_s3_bucket,
                    prefix=prefix,
                )

                if target_step is None:
                    raise ValueError(
                        f"No checkpoints found not after step {not_after_step} for model {from_model} in S3"
                    )

                if verbose:
                    print(
                        f"Found checkpoint at step {target_step}, pulling only this checkpoint..."
                    )

                # Pull only the specific checkpoint we need
                await pull_model_from_s3(
                    model_name=from_model,
                    project=from_project,
                    step=target_step,
                    s3_bucket=from_s3_bucket,
                    verbose=verbose,
                    art_path=self._path,
                    exclude=["logs", "trajectories"],  # Only need checkpoints
                )

        # Find the checkpoint to fork
        checkpoint_base_dir = os.path.join(source_model_dir, "checkpoints")
        if not os.path.exists(checkpoint_base_dir):
            raise FileNotFoundError(
                f"No checkpoints found for model {from_model} in project {from_project}"
            )

        if verbose:
            print(f"DEBUG: Checkpoint base dir: {checkpoint_base_dir}")
            print(
                f"DEBUG: Contents: {os.listdir(checkpoint_base_dir) if os.path.exists(checkpoint_base_dir) else 'Does not exist'}"
            )

        # Get all available checkpoint steps
        available_steps = sorted(
            int(d)
            for d in os.listdir(checkpoint_base_dir)
            if os.path.isdir(os.path.join(checkpoint_base_dir, d)) and d.isdigit()
        )

        if not available_steps:
            raise FileNotFoundError(
                f"No checkpoint directories found for model {from_model}"
            )

        # Determine which step to use
        if not_after_step is None:
            # Use the latest checkpoint
            selected_step = available_steps[-1]
        else:
            # Find the last checkpoint not after the specified step
            valid_steps = [s for s in available_steps if s <= not_after_step]
            if not valid_steps:
                raise ValueError(
                    f"No checkpoints found not after step {not_after_step}. "
                    f"Available steps: {available_steps}"
                )
            selected_step = valid_steps[-1]

        # Create destination checkpoint directory
        dest_checkpoint_dir = get_step_checkpoint_dir(dest_model_dir, selected_step)
        os.makedirs(os.path.dirname(dest_checkpoint_dir), exist_ok=True)

        # Copy the checkpoint
        source_checkpoint_dir = os.path.join(
            checkpoint_base_dir, f"{selected_step:04d}"
        )
        if verbose:
            print(
                f"Copying checkpoint from {source_checkpoint_dir} to {dest_checkpoint_dir}"
            )
            print(f"DEBUG: Source dir exists: {os.path.exists(source_checkpoint_dir)}")
            if os.path.exists(source_checkpoint_dir):
                print(
                    f"DEBUG: Source dir contents: {os.listdir(source_checkpoint_dir)}"
                )
                print(
                    f"DEBUG: Source dir is empty: {len(os.listdir(source_checkpoint_dir)) == 0}"
                )

        import shutil

        # Remove destination if it already exists (empty directory from previous attempts)
        if os.path.exists(dest_checkpoint_dir):
            if verbose:
                print("DEBUG: Destination already exists, removing it first")
            shutil.rmtree(dest_checkpoint_dir)

        shutil.copytree(source_checkpoint_dir, dest_checkpoint_dir)

        if verbose:
            print(
                f"Successfully forked checkpoint from {from_model} (step {selected_step}) to {model.name}"
            )
