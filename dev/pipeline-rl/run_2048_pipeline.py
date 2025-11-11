"""
2048 Game with PipelineRL Concurrent Training

This script demonstrates using PipelineRL's concurrent pipeline for the 2048 game task.

Key differences from baseline:
- Uses concurrent generation and training (not sequential)
- vLLM runs continuously while training happens
- LoRA weights are swapped after each training step
- Expected ~1.5-2x throughput improvement

Requirements:
- 2 GPU machine (GPU 0 for vLLM, GPU 1 for training)
- CUDA_VISIBLE_DEVICES=0,1

Usage:
    CUDA_VISIBLE_DEVICES=0,1 uv run python dev/pipeline-rl/run_2048_pipeline.py
"""

import asyncio
import logging
import math
import os
import random
import string
import xml.etree.ElementTree as ET
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel

import art
from art import TrainableModel, TrainConfig, Trajectory, TrajectoryGroup
from art.dev import (
    InitArgs,
    InternalModelConfig,
    PeftArgs,
    TrainerArgs,
)
from art.dev import (
    TrainConfig as DevTrainConfig,
)
from art.local import LocalBackend
from art.preprocessing.pack import packed_tensors_to_dir
from art.utils.output_dirs import get_model_dir, get_step_checkpoint_dir

# Configure logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

if not root_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

logging.getLogger("art.local.backend").setLevel(logging.INFO)
logging.getLogger("art.local.pipeline_rl_service").setLevel(logging.INFO)

# Disable httpx logging (too verbose with vLLM API calls)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================================================
# 2048 Game Logic (copied from baseline)
# ============================================================================

WINNING_VALUE = 64


class TwentyFortyEightGame(dict):
    """Game state for 2048"""

    id: str
    board: list[list[int | None]]


def populate_random_cell(game: TwentyFortyEightGame) -> None:
    """Randomly populates a cell on the board with a 2 or 4"""
    all_clear_coordinates = [
        (i, j)
        for i in range(len(game["board"]))
        for j in range(len(game["board"][i]))
        if game["board"][i][j] is None
    ]
    random_clear_coordinates = random.choice(all_clear_coordinates)
    # 90% chance to populate a 2, 10% chance to populate a 4
    game["board"][random_clear_coordinates[0]][random_clear_coordinates[1]] = (
        2 if random.random() < 0.9 else 4
    )


def generate_game(board_length: int = 4) -> TwentyFortyEightGame:
    """Generates a new game of 2048"""
    id = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    game = {
        "id": id,
        "board": [[None for _ in range(board_length)] for _ in range(board_length)],
    }

    # populate two random cells
    populate_random_cell(game)
    populate_random_cell(game)

    return game


def render_board(game: TwentyFortyEightGame) -> str:
    """Renders the board in a human-readable format"""
    board = game["board"]
    max_cell_width = max(
        [len(str(cell)) for row in board for cell in row if cell is not None]
    )

    board_str = ""
    for row in board:
        board_str += "|".join(
            [
                str(cell).rjust(max_cell_width)
                if cell is not None
                else "_".rjust(max_cell_width)
                for cell in row
            ]
        )
        board_str += "\n"
    return board_str


def condense_sequence(sequence: list[int | None]) -> list[int | None]:
    """Condense, privileging matches at the start of the sequence"""
    condensed_sequence = []
    gapless_sequence = [cell for cell in sequence if cell is not None]

    i = 0
    while i < len(gapless_sequence):
        if (
            i + 1 < len(gapless_sequence)
            and gapless_sequence[i] == gapless_sequence[i + 1]
        ):
            condensed_sequence.append(gapless_sequence[i] * 2)
            i += 2
        else:
            condensed_sequence.append(gapless_sequence[i])
            i += 1

    # pad the sequence with None at the end
    return condensed_sequence + [None] * (4 - len(condensed_sequence))


def condense_board(
    game: TwentyFortyEightGame, direction: Literal["left", "right", "up", "down"]
) -> None:
    """Condenses the board in a given direction"""
    if direction == "left":
        for row in game["board"]:
            condensed_row = condense_sequence(row)
            for i in range(len(row)):
                row[i] = condensed_row[i]

    if direction == "right":
        for row in game["board"]:
            reversed_row = row[::-1]
            condensed_row = condense_sequence(reversed_row)[::-1]
            for i in range(len(row)):
                row[i] = condensed_row[i]

    if direction == "up":
        for col_index in range(len(game["board"][0])):
            column = [row[col_index] for row in game["board"]]
            condensed_column = condense_sequence(column)
            for row_index in range(len(column)):
                game["board"][row_index][col_index] = condensed_column[row_index]

    if direction == "down":
        for col_index in range(len(game["board"][0])):
            column = [row[col_index] for row in game["board"]]
            reversed_column = column[::-1]
            condensed_column = condense_sequence(reversed_column)[::-1]
            for row_index in range(len(column)):
                game["board"][row_index][col_index] = condensed_column[row_index]


def apply_agent_move(game: TwentyFortyEightGame, move_xml: str) -> None:
    """Applies an agent move to the game board"""
    direction = None
    try:
        root = ET.fromstring(move_xml)
        direction = root.text
    except Exception:
        raise ValueError("Invalid xml")

    if direction not in ["left", "right", "up", "down"]:
        raise ValueError("Invalid direction")

    condense_board(game, direction)
    populate_random_cell(game)


def max_cell_value(game: TwentyFortyEightGame) -> int:
    """Returns the maximum cell value on the board"""
    return max([cell for row in game["board"] for cell in row if cell is not None])


def check_game_finished(game: TwentyFortyEightGame) -> bool:
    """Returns True if the game is finished"""
    if max_cell_value(game) >= WINNING_VALUE:
        return True

    # check if any cell is empty
    if any(cell is None for row in game["board"] for cell in row):
        return False

    return True


def total_board_value(game: TwentyFortyEightGame) -> int:
    """Returns the sum of all the cell values on the board"""
    return sum([cell for row in game["board"] for cell in row if cell is not None])


# ============================================================================
# 2048 Trajectory Generation for PipelineRL
# ============================================================================


class Scenario2048(BaseModel):
    step: int


@art.retry(exceptions=(Exception,))
async def rollout(model: art.Model, scenario: Scenario2048) -> art.Trajectory:
    """
    Generate a single 2048 game trajectory (same as baseline).

    This plays out a full game (multi-turn) and returns the complete trajectory.
    """
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    game = generate_game()
    move_number = 0

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": "You are an excellent 2048 player. Always choose the move most likely to lead to combine cells to eventually reach the number 2048. Optional moves are 'left', 'right', 'up', 'down'. Return your move as an XML object with a single property 'move', like so: <move>left</move>",
            }
        ],
        metadata={
            "game_id": game["id"],
            "notebook-id": "2048",
            "step": scenario.step,
        },
        reward=0,
    )

    while True:
        trajectory.messages_and_choices.append(
            {"role": "user", "content": render_board(game)}
        )

        try:
            messages = trajectory.messages()
            chat_completion = await client.chat.completions.create(
                max_completion_tokens=128,
                messages=messages,
                model=model.get_inference_name(),
                logprobs=True,
            )
        except Exception as e:
            logger.warning(f"Exception generating chat completion: {e}")
            raise e

        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        trajectory.messages_and_choices.append(choice)

        try:
            apply_agent_move(game, content)
            move_number += 1
        except ValueError:
            trajectory.reward = -1
            break

        if check_game_finished(game):
            max_value = max_cell_value(game)
            board_value = total_board_value(game)
            trajectory.metrics["max_value"] = max_value
            trajectory.metrics["board_value"] = board_value
            trajectory.metrics["move_number"] = move_number

            # Compute reward (same as baseline)
            if max_value < WINNING_VALUE:
                max_value_reward = (math.log(max_value, 2) - 1) / (
                    math.log(WINNING_VALUE, 2) - 1
                )
                board_value_reward = (math.log(board_value, 2) - 1) / (
                    math.log(WINNING_VALUE * 16, 2) - 1
                )
                trajectory.reward = max_value_reward + (board_value_reward * 0.2)
            else:
                # Double reward if the agent wins
                trajectory.reward = 2
            break

    return trajectory


# ============================================================================
# Modified Concurrent Pipeline for 2048
# ============================================================================


async def pipeline_rl_train_2048(
    backend: LocalBackend,
    model: TrainableModel,
    config: TrainConfig,
    dev_config: DevTrainConfig,
    inference_gpu_ids: list[int],
    trainer_gpu_ids: list[int],
    num_iterations: int,
    games_per_step: int,
    max_step_lag: int,
    base_url: str,
):
    """
    Custom PipelineRL training loop for 2048 game.

    This is similar to _pipeline_rl_train_concurrent but uses custom 2048 generation.
    """
    logger.info("[2048_PIPELINE] Starting concurrent training with 2048 game...")

    # Start vLLM and initialize process group (same as standard pipeline)
    # from art.local.pipeline_rl_service import PipelineRLService

    # assert isinstance(service, PipelineRLService), "Model must use PipelineRLService"

    # # Initialize process group
    # init_method, world_size = await backend._init_process_group_for_weight_updates(
    #     model, len(inference_gpu_ids)
    # )
    #
    # original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    #
    # # Set CUDA_VISIBLE_DEVICES to only show trainer GPUs
    # # This will be inherited by the subprocess created by @mp_actors.move
    # trainer_gpu_str = ",".join(str(gpu_id) for gpu_id in trainer_gpu_ids)
    # os.environ["CUDA_VISIBLE_DEVICES"] = trainer_gpu_str
    # logger.info(
    #     f"[PIPELINE_RL]   Temporarily set CUDA_VISIBLE_DEVICES={trainer_gpu_str}"
    # )
    # logger.info(f"[PIPELINE_RL]   (original was: {original_cuda_visible})")
    #
    # # Get service
    # try:
    #     service = await backend._get_service(model)
    #     logger.info(f"Service: {service._obj}")
    # finally:
    #     if original_cuda_visible is not None:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
    #     else:
    #         os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    #     logger.info(
    #         f"[PIPELINE_RL]   Restored CUDA_VISIBLE_DEVICES to: {original_cuda_visible}"
    #     )
    #
    # # Start vLLM
    # logger.info("[2048_PIPELINE] Starting vLLM...")
    # await service.start_openai_server_with_weight_updates(
    #     config=dev_config.get("openai_server_config", None),
    #     init_method=init_method,
    #     world_size=world_size,
    #     actor_idx=0,
    #     inference_gpu_ids=inference_gpu_ids,
    # )
    #
    # # Join process group
    # backend._actor_update_group = await backend._join_process_group_as_trainer(
    #     init_method, world_size
    # )
    #
    # # Wait for vLLM
    # await backend._wait_for_vllm_ready(base_url)
    # logger.info("[2048_PIPELINE] vLLM is ready!")
    #
    # Create queues
    trajectory_queue = asyncio.Queue(maxsize=50)
    metrics_queue = asyncio.Queue()

    # Generation task
    async def generation_task():
        for step in range(num_iterations):
            logger.info(f"[GENERATION] Starting generation step {step}")

            try:
                # Generate 2048 game trajectories using standard ART pattern
                trajectory_groups = await art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(
                            rollout(model, Scenario2048(step=step))
                            for _ in range(games_per_step)
                        )
                        for _ in range(1)
                    ),
                    pbar_desc=f"generate_step_{step}",
                    max_exceptions=games_per_step,
                )

                logger.info(
                    f"[GENERATION] Putting {len(trajectory_groups)} groups in queue"
                )
                await trajectory_queue.put(trajectory_groups)

                logger.info(f"[GENERATION] Completed generation step {step}")

            except Exception as e:
                logger.error(f"[GENERATION] Error in step {step}: {e}")
                raise

        # Signal end
        await trajectory_queue.put(None)
        logger.info("[GENERATION] Generation task complete")

    # Training task
    async def training_task():
        training_step = 0

        while True:
            logger.info(
                f"[TRAINING] Waiting for trajectories (queue size: {trajectory_queue.qsize()}/50)"
            )
            trajectory_groups = await trajectory_queue.get()

            if trajectory_groups is None:
                logger.info("[TRAINING] Received end signal")
                break

            logger.info(f"[TRAINING] Starting training step {training_step}")
            logger.info(
                f"[TRAINING]   Received {len(trajectory_groups)} trajectory groups"
            )

            try:
                # Count submitted groups and trainable groups
                #                 num_groups_submitted = len(trajectory_groups)
                #                 num_groups_trainable = sum(
                #                     1
                #                     for group in trajectory_groups
                #                     if group and len(set(trajectory.reward for trajectory in group)) > 1
                #                 )
                #                 logger.info(
                #                     f"[TRAINING] {num_groups_submitted} groups, {num_groups_trainable} groups trainable"
                #                 )
                #
                #                 # Pack trajectories (reuse existing ART functionality)
                #                 packed_tensors = backend._get_packed_tensors(
                #                     model,
                #                     trajectory_groups,
                #                     advantage_balance=dev_config.get("advantage_balance", 0.0),
                #                     allow_training_without_logprobs=dev_config.get(
                #                         "allow_training_without_logprobs", False
                #                     ),
                #                     scale_rewards=dev_config.get("scale_rewards", True),
                #                     plot_tensors=dev_config.get("plot_tensors", False),
                #                 )
                #
                #                 if packed_tensors is None:
                #                     logger.warning("[TRAINING] No suitable training data, skipping...")
                #                     trajectory_queue.task_done()
                #                     continue
                #
                #                 disk_packed_tensors = packed_tensors_to_dir(
                #                     packed_tensors,
                #                     f"{get_model_dir(model=model, art_path=backend._path)}/tensors",
                #                 )
                #
                #                 # Train
                #                 logger.info(f"[TRAINING] Training on batch...")
                # #
                #                 async for metrics in service.train(
                #                     disk_packed_tensors, config, dev_config
                #                 ):
                #                     logger.info(f"[TRAINING]   Metrics: {metrics}")
                #                     await metrics_queue.put(metrics)

                await model.train(trajectory_groups, config=config, _config=dev_config)

                # Swap LoRA checkpoint
                checkpoint_dir = get_step_checkpoint_dir(
                    get_model_dir(model=model, art_path=backend._path),
                    await model.get_step(),
                )
                logger.info(f"[TRAINING] Swapping LoRA to {checkpoint_dir}")
                await service.swap_lora_checkpoint(checkpoint_dir, base_url)

                training_step += 1
                logger.info(f"[TRAINING] Completed training step {training_step}")

                trajectory_queue.task_done()

            except Exception as e:
                logger.error(f"[TRAINING] Error in step {training_step}: {e}")
                raise

        # Signal end
        await metrics_queue.put(None)
        logger.info("[TRAINING] Training task complete")

    # Launch both tasks
    logger.info("[2048_PIPELINE] Launching concurrent tasks...")
    gen_task = asyncio.create_task(generation_task())
    train_task = asyncio.create_task(training_task())

    # Yield metrics as they arrive
    while True:
        metrics = await metrics_queue.get()
        if metrics is None:
            break
        yield metrics

    # Wait for completion
    await asyncio.gather(gen_task, train_task)

    logger.info("[2048_PIPELINE] Concurrent training complete!")


# ============================================================================
# Main Script
# ============================================================================


async def main():
    logger.info("=" * 80)
    logger.info("2048 Game with PipelineRL Concurrent Training")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This demonstrates PipelineRL with the 2048 game task:")
    logger.info("  - Concurrent generation and training")
    logger.info("  - Multi-turn game trajectories")
    logger.info("  - Real vLLM inference")
    logger.info("  - LoRA weight swapping after each step")
    logger.info("")

    random.seed(42)

    # Initialize backend
    backend = LocalBackend()

    # Create trainable model with PipelineRL config
    logger.info("[1] Creating trainable model with PipelineRL configuration...")
    model = TrainableModel(
        name="agent-001",
        project="2048-pipeline",
        base_model="OpenPipe/Qwen3-14B-Instruct",
        _internal_config=InternalModelConfig(
            _use_pipeline_rl=True,
            # init_args=InitArgs(
            #     max_seq_length=2048,
            #     load_in_4bit=False,
            #     load_in_8bit=False,
            #     fast_inference=False,
            # ),
            # peft_args=PeftArgs(r=8, lora_alpha=32),
            # trainer_args=TrainerArgs(
            #     output_dir=".art/2048-pipeline",
            #     per_device_train_batch_size=2,
            #     gradient_accumulation_steps=1,
            #     num_train_epochs=1,
            # ),
        ),
    )

    # Register model
    logger.info("[2] Registering model...")
    await model.register(backend)
    logger.info(f"MODEL inference URL: {model.inference_base_url}")
    logger.info(f"MODEL inference API Key: {model.inference_api_key}")

    # # Training configuration
    # train_config = TrainConfig(
    #     learning_rate=1e-5,
    #     beta=0.1,
    # )
    #
    # dev_config = DevTrainConfig(
    #     allow_training_without_logprobs=False,
    # )

    # Configure GPU assignment
    trainer_gpu_ids = [0]  # Unsloth uses GPU 0
    inference_gpu_ids = [1]  # vLLM uses GPU 1

    # Pipeline parameters
    num_iterations = 20  # Run for 20 training steps
    games_per_step = 18  # Generate 18 games per step (same as baseline)
    max_step_lag = 5  # Allow generation to get 5 steps ahead

    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  - num_iterations: {num_iterations} training steps")
    logger.info(f"  - games_per_step: {games_per_step} games per step")
    logger.info(f"  - max_step_lag: {max_step_lag} (queue-based backpressure)")
    logger.info(f"  - WINNING_VALUE: {WINNING_VALUE}")

    # Run PipelineRL training
    logger.info("\n[3] Starting PipelineRL concurrent training...")
    logger.info("    GPU Configuration:")
    logger.info(f"      - Inference GPUs (vLLM): {inference_gpu_ids}")
    logger.info(f"      - Training GPUs (Unsloth): {trainer_gpu_ids}")
    logger.info("")

    iteration = 0
    async for metrics in pipeline_rl_train_2048(
        backend=backend,
        model=model,
        config=TrainConfig(),
        dev_config=DevTrainConfig(),
        inference_gpu_ids=inference_gpu_ids,
        trainer_gpu_ids=trainer_gpu_ids,
        num_iterations=num_iterations,
        games_per_step=games_per_step,
        max_step_lag=max_step_lag,
        base_url="http://localhost:8000",
    ):
        logger.info(f"[Main] Training metrics: {metrics}")
        iteration += 1

    logger.info("\n[4] Training complete!")
    logger.info("=" * 80)
    logger.info("✓ Success! 2048 PipelineRL training completed.")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Completed {iteration} training steps with concurrent generation")
    logger.info("")


if __name__ == "__main__":
    asyncio.run(main())
