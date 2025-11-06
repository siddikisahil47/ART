"""
PipelineRL Phase 1: Sequential Pipeline Test

This script tests the PipelineRL sequential pipeline infrastructure:
- Process group initialization
- vLLM startup with weight update support
- Training loop with LoRA swapping
- Sequential iteration: train → swap LoRA → repeat

Phase 1 uses dummy trajectories to focus on testing infrastructure.
Phase 2 will add real trajectory generation with concurrent execution.

Requirements:
- 2 GPU machine (GPU 0 for vLLM, GPU 1 for training)
- CUDA_VISIBLE_DEVICES=0,1

Usage:
    CUDA_VISIBLE_DEVICES=0,1 uv run python dev/pipeline-rl/run_sequential.py
"""

import asyncio
import logging

import art
from art import Trajectory, TrajectoryGroup, TrainableModel, TrainConfig
from art.local import LocalBackend
from art.dev import (
    InternalModelConfig,
    TrainConfig as DevTrainConfig,
    InitArgs,
    PeftArgs,
    TrainerArgs,
    OpenAIServerConfig,
)

# Configure logging to show all output
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

# Explicitly set levels for ART loggers
logging.getLogger("art.local.backend").setLevel(logging.INFO)
logging.getLogger("art.local.pipeline_rl_service").setLevel(logging.INFO)

# Get logger for this script
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_dummy_trajectory_groups(
    iteration: int,
    num_groups: int = 20,
    trajectories_per_group: int = 2,
) -> list[TrajectoryGroup]:
    """
    Create dummy trajectory groups for testing infrastructure.

    These have proper structure (user + assistant messages with different rewards)
    to pass training data validation, but use dummy content.

    Args:
        iteration: Current iteration number
        num_groups: Number of trajectory groups to create
        trajectories_per_group: Number of trajectories per group

    Returns:
        List of TrajectoryGroup objects
    """
    trajectory_groups = []

    for group_idx in range(num_groups):
        trajectories = []

        for traj_idx in range(trajectories_per_group):
            # Create user message
            user_message = {
                "role": "user",
                "content": f"Prompt {iteration}-{group_idx}-{traj_idx}: Respond with yes, no, or maybe",
            }

            # Create assistant message (response)
            responses = ["yes", "no", "maybe"]
            response_content = responses[traj_idx % len(responses)]
            assistant_message = {
                "role": "assistant",
                "content": response_content,
            }

            # Assign rewards (different for each trajectory to enable ranking)
            rewards = [0.5, 0.75, 1.0]  # Different rewards for ranking
            reward = rewards[traj_idx % len(rewards)]

            # Create trajectory with messages_and_choices
            trajectory = Trajectory(
                messages_and_choices=[user_message, assistant_message],
                reward=reward,
                metadata={
                    "iteration": iteration,
                    "group": group_idx,
                    "trajectory": traj_idx,
                },
            )
            trajectories.append(trajectory)

        # Create TrajectoryGroup
        trajectory_group = TrajectoryGroup(trajectories)
        trajectory_groups.append(trajectory_group)

    logger.info(
        f"  Created {len(trajectory_groups)} trajectory groups for iteration {iteration}"
    )
    return trajectory_groups


async def main():
    logger.info("=" * 80)
    logger.info("PipelineRL Phase 1: Sequential Pipeline Infrastructure Test")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This test focuses on infrastructure, not the task:")
    logger.info("  - Process group initialization")
    logger.info("  - vLLM with weight update support")
    logger.info("  - Training loop")
    logger.info("  - LoRA checkpoint swapping")
    logger.info("")

    # Initialize backend
    backend = LocalBackend()

    # Create trainable model with PipelineRL config
    logger.info("[1] Creating trainable model with PipelineRL configuration...")
    model = TrainableModel(
        name="pipeline-rl-test",
        project="pipelinerl-infrastructure",
        base_model="OpenPipe/Qwen3-14B-Instruct",
        _internal_config=InternalModelConfig(
            _use_pipeline_rl=True,  # Enable PipelineRLService
            init_args=InitArgs(
                max_seq_length=1024,  # AssertionError: Sequence length (512) must be evenly divisible by chunk size (1024)
                load_in_4bit=False,
                load_in_8bit=False,
                fast_inference=False,  # Don't load vLLM via Unsloth (we manage it separately)
            ),
            peft_args=PeftArgs(r=1, lora_alpha=32),
            trainer_args=TrainerArgs(
                output_dir=".art/pipeline-rl-test",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                num_train_epochs=1,
            ),
        ),
    )

    # Register model
    logger.info("[2] Registering model...")
    await backend.register(model)

    # Generate dummy trajectory groups for testing
    num_iterations = 3
    logger.info(
        f"\n[3] Generating dummy trajectory groups ({num_iterations} iterations)..."
    )
    logger.info("    Using dummy data to focus on infrastructure testing")

    trajectory_groups_list = []
    for iteration in range(num_iterations):
        trajectory_groups = create_dummy_trajectory_groups(
            iteration=iteration,
            num_groups=20,  # 20 groups per iteration
            trajectories_per_group=2,  # 2 trajectories per group for ranking
        )
        trajectory_groups_list.append(trajectory_groups)

    logger.info(f"    ✓ Generated trajectory groups for {num_iterations} iterations")

    # Training configuration
    train_config = TrainConfig(
        learning_rate=1e-4,
        beta=0.1,  # KL penalty coefficient
    )

    dev_config = DevTrainConfig(
        allow_training_without_logprobs=True, # there is no logprobs since we are using dummy trajactories
    )

    # Run PipelineRL training (sequential)
    logger.info("\n[4] Starting PipelineRL sequential training...")
    logger.info("    This will:")
    logger.info("      - Start vLLM with process group support")
    logger.info("      - For each iteration:")
    logger.info("        * Train on trajectories")
    logger.info("        * Save LoRA checkpoint")
    logger.info("        * Swap LoRA in vLLM via HTTP API")
    logger.info("")

    iteration = 0
    async for metrics in backend._pipeline_rl_train(
        model=model,
        trajectory_groups_list=trajectory_groups_list,
        config=train_config,
        dev_config=dev_config,
        num_actor_gpus=1,  # Single GPU for vLLM (GPU 0)
        max_step_lag=5,
        base_url="http://localhost:8000",
    ):
        logger.info(f"[Iteration {iteration}] Training metrics: {metrics}")
        iteration += 1

    logger.info("\n[5] Training complete!")
    logger.info("=" * 80)
    logger.info("✓ Success! PipelineRL Phase 1 sequential training completed.")
    logger.info("=" * 80)
    logger.info("")
    logger.info("What we tested:")
    logger.info("  ✓ Process group initialization (tcp:// with NCCL)")
    logger.info("  ✓ vLLM startup with weight update support")
    logger.info("  ✓ Sequential training loop")
    logger.info("  ✓ LoRA checkpoint saving")
    logger.info("  ✓ LoRA swapping via vLLM HTTP API")
    logger.info("  ✓ Step tracking (generation_step, training_step, lag)")
    logger.info("")
    logger.info("Phase 2 will add:")
    logger.info("  → Concurrent generation during training")
    logger.info("  → In-flight weight updates via NCCL broadcast")
    logger.info("  → Real trajectory generation with vLLM")
    logger.info("  → Backpressure with max_step_lag")


if __name__ == "__main__":
    asyncio.run(main())
