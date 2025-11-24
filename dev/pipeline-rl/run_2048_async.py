"""
2048 Game with PipelineRL Async Training

- Implements continous batching

Requirements:
- 2 GPU machine (GPU 0 for vLLM, GPU 1 for training)
- CUDA_VISIBLE_DEVICES=0,1

Usage:
    CUDA_VISIBLE_DEVICES=0,1 uv run python dev/pipeline-rl/run_2048_async.py
"""

import argparse
import asyncio
import logging
import math
import os
import random
import string
import xml.etree.ElementTree as ET
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

import art
from art import TrainableModel, TrainConfig, Trajectory, TrajectoryGroup
from art.dev import (
    InternalModelConfig,
)
from art.dev.engine import EngineArgs
from art.local import LocalBackend

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

def calculate_group_std_dev(trajectory_group: TrajectoryGroup) -> float:
    rewards = [t.reward for t in trajectory_group.trajectories]

    if len(rewards) > 1:
        return np.std(rewards)
    else:
        return 0


async def main(
    num_steps: int,
    rollouts_per_group: int,
    groups_per_step: int,
    max_concurrent_rollouts: int,
):
    load_dotenv()
    logger.info("=" * 80)
    logger.info("2048 Game with AsyncService")
    logger.info("=" * 80)

    random.seed(42)

    # Show current CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
    logger.info(f"Current CUDA_VISIBLE_DEVICES: {cuda_visible}")
    logger.info("")

    # Declare the model
    logger.info("[1] Creating TrainableModel with AsyncService configuration...")
    model = art.TrainableModel(
        name="agent-002",
        project="pipeline-rl-test",
        base_model="OpenPipe/Qwen3-14B-Instruct",
        _internal_config=InternalModelConfig(
            engine_args=EngineArgs(
                max_model_len=512,
            ),
            _async_rl=True,
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
        ),
    )
    logger.info(f"    Model name: {model.name}")
    logger.info(f"    Base model: {model.base_model}")
    logger.info(
        f"    Trainer GPUs: {model._internal_config.get('trainer_gpu_ids', [0])}"
    )
    logger.info(
        f"    Inference GPUs: {model._internal_config.get('inference_gpu_ids', [1])}"
    )
    logger.info("")

    # Initialize the backend
    logger.info("[2] Initializing LocalBackend...")
    backend = LocalBackend()
    logger.info("    Backend initialized")
    logger.info("")

    # Register model (this triggers service initialization)
    logger.info("[3] Registering model with backend...")
    logger.info("    This will:")
    logger.info("      - Create AsyncService")
    logger.info("      - Initialize trainer on GPU 0")
    logger.info("      - Start vLLM server on GPU 1")
    logger.info("")

    await model.register(backend)

    logger.info("")
    logger.info("[4] Model registration complete!")
    logger.info("")
    logger.info("Service Status:")
    logger.info("  - Trainer should be running on GPU 0")
    logger.info("  - vLLM server should be running on GPU 1")
    logger.info("")
    logger.info("Verify with: nvidia-smi")
    logger.info("")
    logger.info(
        f"Configuration: num_steps={num_steps}, rollouts_per_group={rollouts_per_group}, groups_per_step={groups_per_step}, max_concurrent_rollouts={max_concurrent_rollouts}"
    )

    trajectory_queue = asyncio.Queue(maxsize=50)
    metrics_queue = asyncio.Queue()

    # Generation task
    async def generation_task():
        active_rollouts: set[asyncio.Task] = set()

        # Buffer to reassemble groups: { group_id: [trajectory1, trajectory2, ...] }
        group_buffer: dict[int, list[art.Trajectory]] = {}

        total_groups_needed = num_steps * groups_per_step
        total_rollouts_needed = total_groups_needed * rollouts_per_group

        # We track progress by rollouts launched
        rollouts_launched = 0

        logger.info(f"[GENERATION] Starting pool of {max_concurrent_rollouts} rollouts")

        while rollouts_launched < total_rollouts_needed or active_rollouts:
            # 1. Refill the pool with individual rollouts
            while (
                len(active_rollouts) < max_concurrent_rollouts
                and rollouts_launched < total_rollouts_needed
            ):
                # Calculate which group this rollout belongs to
                # Group ID is global (0 to total_groups_needed-1)
                current_group_id = rollouts_launched // rollouts_per_group
                current_step_idx = current_group_id // groups_per_step

                # Define a wrapper to attach metadata (group_id) to the task result
                async def rollout_worker(gid, step):
                    traj = await rollout(model, Scenario2048(step=step))
                    return gid, traj

                task = asyncio.create_task(
                    rollout_worker(current_group_id, current_step_idx)
                )
                active_rollouts.add(task)
                rollouts_launched += 1

            if rollouts_launched % 50 == 0:  # Log every 50 rollouts
                logger.info(
                    f"[GENERATION] Progress: {rollouts_launched}/{total_rollouts_needed} "
                    f"({len(active_rollouts)} active, {len(group_buffer)} partial groups)"
                )

            # 2. Wait for FIRST rollout to finish
            done, _ = await asyncio.wait(
                active_rollouts, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                active_rollouts.remove(task)
                try:
                    # Get result: (group_id, trajectory)
                    gid, traj = await task

                    # Add to buffer
                    if gid not in group_buffer:
                        group_buffer[gid] = []
                    group_buffer[gid].append(traj)

                    logger.info(
                        f"[GENERATION] Group {gid} {len(group_buffer[gid])}/{rollouts_per_group} completed"
                    )

                    # Check if group is complete
                    if len(group_buffer[gid]) == rollouts_per_group:
                        # Create the TrajectoryGroup
                        complete_group = art.TrajectoryGroup(group_buffer.pop(gid))
                        logger.info(
                            f"[GENERATION] Group {gid} completed, sending to trainer "
                            f"(queue size: {trajectory_queue.qsize()})"
                        )

                        # Send to trainer
                        await trajectory_queue.put(complete_group)

                except Exception as e:
                    logger.error(f"[GENERATION] Rollout failed: {e}")
                    raise e

        # Signal end
        await trajectory_queue.put(None)
        logger.info("[GENERATION] Generation task complete")

    # Training task
    async def training_task():
        training_step = 0
        batch_buffer = []

        while True:
            logger.info(
                f"[TRAINING] Waiting for trajectories (queue size: {trajectory_queue.qsize()}/50)"
            )
            trajectory_group = await trajectory_queue.get()

            if trajectory_group is None:
                logger.info("[TRAINING] Received end signal")
                break

            if calculate_group_std_dev(trajectory_group) != 0:
                batch_buffer.append(trajectory_group)
                logger.info(
                    f"[TRAINING]: Group received {len(batch_buffer)}/{groups_per_step} received"
                )
            else:
                logger.info(
                    f"[TRAINING]: Group received but skipped (reward stdev = 0) {len(batch_buffer)}/{groups_per_step} received"
                )

            if len(batch_buffer) >= groups_per_step:
                logger.info(f"[TRAINING] Starting training step {training_step}")
                logger.info(
                    f"[TRAINING]   Received {len(batch_buffer)} trajectory groups"
                )

                try:
                    await model.train(
                        batch_buffer,
                        config=art.TrainConfig(),
                        _config=art.dev.TrainConfig(),
                    )

                    logger.info(f"[TRAINING] Completed training step {training_step}")
                    training_step += 1

                    # Mark all items as done
                    for _ in range(len(batch_buffer)):
                        trajectory_queue.task_done()
                    batch_buffer = []

                except Exception as e:
                    logger.error(f"[TRAINING] Error in step {training_step}: {e}")
                    raise

        # Signal end
        await metrics_queue.put(None)
        logger.info("[TRAINING] Training task complete")

    gen_task = asyncio.create_task(generation_task())
    train_task = asyncio.create_task(training_task())

    # Yield metrics as they arrive
    while True:
        metrics = await metrics_queue.get()
        if metrics is None:
            break
        logger.info(f"Training metrics: {metrics}")

    # Wait for completion
    await asyncio.gather(gen_task, train_task)
    logger.info("Press Ctrl+C to stop...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AsyncService Test Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of training steps to run",
    )
    parser.add_argument(
        "--rollouts-per-group",
        type=int,
        default=18,
        help="Number of rollouts per trajectory group",
    )
    parser.add_argument(
        "--groups-per-step",
        type=int,
        default=1,
        help="Number of trajectory groups per step",
    )
    parser.add_argument(
        "--sleep-per-step",
        type=int,
        default=0,
        help="Amount of time (s) to wait between each step",
    )
    parser.add_argument(
        "--max-concurrent-rollouts",
        type=int,
        default=18,
        help="Maximum concurrent rollouts",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            args.num_steps,
            args.rollouts_per_group,
            args.groups_per_step,
            args.max_concurrent_rollouts,
        )
    )
