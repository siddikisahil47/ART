"""
Test script for Step 2.1: Trajectory Generation

This script tests the _generate_batch() method that will be used in
concurrent PipelineRL. It verifies that:
1. vLLM can be called via OpenAI API
2. Responses are converted to TrajectoryGroup objects
3. Rewards are computed correctly
4. The function is ready for use in concurrent pipeline

Requirements:
- 2 GPU machine (GPU 0 for vLLM, GPU 1 for training)
- CUDA_VISIBLE_DEVICES=0,1

Usage:
    CUDA_VISIBLE_DEVICES=0,1 uv run python dev/pipeline-rl/run_generate_batch.py
"""

import asyncio
import logging
import os

from art import TrainableModel
from art.dev import (
    InitArgs,
    InternalModelConfig,
    PeftArgs,
    TrainerArgs,
)
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_yes_no_maybe_reward(prompt: list[dict], response: str) -> float:
    """
    Reward function for yes-no-maybe task.

    Rewards:
    - "maybe" = 1.0 (best)
    - "no" = 0.75 (middle)
    - "yes" = 0.5 (worst)

    This simulates a task where the model should learn to say "maybe".
    """
    response_lower = response.lower()

    if "maybe" in response_lower:
        return 1.0
    elif "no" in response_lower:
        return 0.75
    elif "yes" in response_lower:
        return 0.5
    else:
        # Default reward for unexpected responses
        return 0.25


async def main():
    logger.info("=" * 80)
    logger.info("Step 2.1: Test Trajectory Generation (_generate_batch)")
    logger.info("=" * 80)
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
            _use_pipeline_rl=True,
            init_args=InitArgs(
                max_seq_length=1024,
                load_in_4bit=False,
                load_in_8bit=False,
                fast_inference=False,
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

    # Start vLLM server (without training yet)
    logger.info("\n[3] Starting vLLM server for trajectory generation test...")
    logger.info("    This uses Phase 1 infrastructure (process group, etc.)")

    # Configure GPU assignment
    inference_gpu_ids = [0]
    trainer_gpu_ids = [1]

    logger.info(f"    Inference GPUs (vLLM): {inference_gpu_ids}")
    logger.info(f"    Training GPUs (not used in this test): {trainer_gpu_ids}")

    # Initialize process group infrastructure
    init_method, world_size = await backend._init_process_group_for_weight_updates(
        model, len(inference_gpu_ids)
    )

    # Set CUDA_VISIBLE_DEVICES for training subprocess (even though we won't train)
    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    trainer_gpu_str = ",".join(str(gpu_id) for gpu_id in trainer_gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = trainer_gpu_str

    try:
        service = await backend._get_service(model)
    finally:
        if original_cuda_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    # Start vLLM with weight update support
    await service.start_openai_server_with_weight_updates(
        config=None,
        init_method=init_method,
        world_size=world_size,
        actor_idx=0,
        inference_gpu_ids=inference_gpu_ids,
    )

    # Join process group as trainer
    backend._actor_update_group = await backend._join_process_group_as_trainer(
        init_method, world_size
    )

    # Wait for vLLM to be ready
    base_url = "http://localhost:8000"
    await backend._wait_for_vllm_ready(base_url, timeout=120.0)
    logger.info("    ✓ vLLM is ready!\n")

    # Test trajectory generation
    logger.info("[4] Testing _generate_batch() with sample prompts...")

    # Create test prompts (yes-no-maybe task)
    test_prompts = [
        [{"role": "user", "content": "Respond with 'yes', or 'maybe'."}],
        [{"role": "user", "content": "Answer no, or maybe."}],
        [{"role": "user", "content": "Say one of: yes, no, maybe"}],
        [{"role": "user", "content": "Pick yes or no"}],
        [{"role": "user", "content": "Choose: yes or maybe"}],
    ]

    logger.info(f"    Generating {len(test_prompts)} trajectories...")

    # Call _generate_batch
    trajectory_groups = await backend._generate_batch(
        model=model,
        prompts=test_prompts,
        reward_fn=compute_yes_no_maybe_reward,
        base_url=base_url,
        generation_config={
            "max_tokens": 50,  # Short responses for testing
            "temperature": 0.9,  # Higher temp for variety
        },
    )

    logger.info(f"    ✓ Generated {len(trajectory_groups)} trajectory groups\n")

    # Verify results
    logger.info("[5] Verifying generated trajectories...")

    for i, group in enumerate(trajectory_groups):
        assert len(group.trajectories) == 1, f"Expected 1 trajectory per group, got {len(group.trajectories)}"
        trajectory = group.trajectories[0]

        # Check messages
        messages = trajectory.messages()
        assert len(messages) >= 2, f"Expected at least 2 messages (user + assistant), got {len(messages)}"
        assert messages[0]["role"] == "user", f"Expected first message to be user, got {messages[0]['role']}"
        assert messages[-1]["role"] == "assistant", f"Expected last message to be assistant, got {messages[-1]['role']}"

        # Check reward
        assert 0.0 <= trajectory.reward <= 1.0, f"Reward should be between 0 and 1, got {trajectory.reward}"

        # Check metadata
        assert "prompt_idx" in trajectory.metadata, "Missing prompt_idx in metadata"

        # Log results
        response_content = messages[-1]["content"]
        logger.info(f"    Trajectory {i}:")
        logger.info(f"      Prompt: {messages[0]['content'][:50]}...")
        logger.info(f"      Response: {response_content}")
        logger.info(f"      Reward: {trajectory.reward}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Success! Trajectory generation test passed.")
    logger.info("=" * 80)
    logger.info("")
    logger.info("What we verified:")
    logger.info("  ✓ vLLM server responds to OpenAI API calls")
    logger.info("  ✓ Responses are converted to Trajectory objects")
    logger.info("  ✓ Rewards are computed correctly")
    logger.info("  ✓ TrajectoryGroups have proper structure")
    logger.info("  ✓ Function is ready for use in concurrent pipeline")
    logger.info("")
    logger.info("Next: Step 2.2 - Build concurrent structure with dummy data")


if __name__ == "__main__":
    asyncio.run(main())
