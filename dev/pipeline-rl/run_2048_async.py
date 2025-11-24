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

import argparse
import asyncio
import logging
import math
import os
import random
import string
import xml.etree.ElementTree as ET
from typing import Literal

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


async def main(
    num_steps: int, rollouts_per_group: int, groups_per_step: int, sleep_per_step: int
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
        name="agent-001",
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
    logger.info(f"Configuration: num_steps={num_steps}, rollouts_per_group={rollouts_per_group}, groups_per_step={groups_per_step}")

    trajectory_queue = asyncio.Queue(maxsize=50)
    metrics_queue = asyncio.Queue()

    # Generation task
    async def generation_task():
        for step in range(num_steps):
            logger.info(f"[GENERATION] Starting generation step {step}")

            try:
                # Generate 2048 game trajectories using standard ART pattern
                trajectory_groups = await art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(
                            rollout(model, Scenario2048(step=step))
                            for _ in range(rollouts_per_group)
                        )
                        for i in range(groups_per_step)
                    ),
                    pbar_desc=f"generate_step_{step}",
                    max_exceptions=rollouts_per_group * groups_per_step,
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
                await model.train(
                    trajectory_groups,
                    config=art.TrainConfig(),
                    _config=art.dev.TrainConfig(),
                )

                training_step += 1
                logger.info(f"[TRAINING] Completed training step {training_step}")

                trajectory_queue.task_done()

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
    args = parser.parse_args()
    asyncio.run(
        main(
            args.num_steps,
            args.rollouts_per_group,
            args.groups_per_step,
            args.sleep_per_step,
        )
    )
