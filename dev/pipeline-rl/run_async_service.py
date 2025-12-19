"""
AsyncService Test Script

This script tests the AsyncService initialization:
- Trainer on GPU 0
- vLLM inference server on GPU 1
- Service initialization and vLLM startup

Usage:
    CUDA_VISIBLE_DEVICES=0,1 uv run python dev/pipeline-rl/run_async_service.py
"""

import argparse
import asyncio
import logging
import os
from time import sleep

from openai import AsyncOpenAI
from pydantic import BaseModel

import art
from art.dev.model import InternalModelConfig
from art.local.backend import LocalBackend

# Configure logging to show all output
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

if not root_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

# Explicitly set levels for ART loggers
logging.getLogger("art.local.backend").setLevel(logging.INFO)
logging.getLogger("art.local.pipeline_rl_service").setLevel(logging.INFO)
logging.getLogger("art.unsloth").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


class ReverseStringScenario(BaseModel):
    input: str


class TrainingScenario(BaseModel):
    step: int
    data: ReverseStringScenario


async def rollout(
    model: art.TrainableModel, scenario: TrainingScenario
) -> art.Trajectory:
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": "You are a string reverser. You will be given a string and you will need to reverse it. Return the reversed string.",
            }
        ],
        metadata={
            "step": scenario.step,
        },
        reward=0,
    )
    trajectory.messages_and_choices.append(
        {
            "role": "user",
            "content": scenario.data.input,
        }
    )
    response = await client.chat.completions.create(
        messages=trajectory.messages(),
        model=model.get_inference_name(),
        logprobs=True,
    )
    trajectory.messages_and_choices.append(response.choices[0])
    if response.choices[0].message.content == scenario.data.input[::-1]:
        trajectory.reward = 1.0
    else:
        trajectory.reward = 0.0
    return trajectory


async def main(
    num_steps: int, rollouts_per_group: int, groups_per_step: int, sleep_per_step: int
):
    logger.info("=" * 80)
    logger.info("AsyncService Test")
    logger.info("=" * 80)
    logger.info("")

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
        f"Configuration: num_steps={num_steps}, rollouts_per_group={rollouts_per_group}, groups_per_step={groups_per_step}"
    )
    logger.info("")
    for step in range(num_steps):
        sleep(sleep_per_step)
        logger.info(f"[GENERATION] Starting generation step {step}")
        import random
        import string

        def random_string(length: int) -> str:
            return "".join(
                random.choices(string.ascii_letters + string.digits, k=length)
            )

        # Generate a list of long random input strings
        random_inputs = [random_string(64) for _ in range(groups_per_step)]
        try:
            trajectory_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        rollout(
                            model,
                            TrainingScenario(
                                step=step,
                                data=ReverseStringScenario(input=random_inputs[i]),
                            ),
                        )
                        for _ in range(rollouts_per_group)
                    )
                    for i in range(groups_per_step)
                ),
                pbar_desc=f"generate_step_{step}",
                max_exceptions=1,
            )

            logger.info(
                f"[GENERATION] Putting {len(trajectory_groups)} groups in queue"
            )
        except Exception as e:
            logger.error(f"[GENERATION] Error in step {step}: {e}")

        logger.info(f"[TRAINING] Starting training step {step}")
        logger.info(f"[TRAINING]   Received {len(trajectory_groups)} trajectory groups")
        try:
            await model.train(
                trajectory_groups,
                config=art.TrainConfig(),
                _config=art.dev.TrainConfig(),
            )
        except Exception as e:
            logger.error(f"[TRAINING] Error in step {step}: {e}")
            raise
        logger.info(f"[TRAINING] Completed training step {step}")

    logger.info("Press Ctrl+C to stop...")

    # Keep the script running so we can see the processes
    try:
        await asyncio.sleep(3600)  # Sleep for 1 hour (or until interrupted)
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Shutting down...")
        await backend.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AsyncService Test Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of training steps to run",
    )
    parser.add_argument(
        "--rollouts-per-group",
        type=int,
        default=1,
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
