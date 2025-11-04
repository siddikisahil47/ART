"""
Tests for LocalBackend process group helper methods.

These tests verify that the process group infrastructure for PipelineRL
weight updates is set up correctly.

NOTE: test_basic_process_group_creation requires a GPU machine to run since
it tests real torch.distributed process group initialization with NCCL backend.
"""

import logging
import socket

# Add src to path for imports
import sys
import threading
from pathlib import Path

import pytest
import torch
import torch.distributed

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import art
from art.local.backend import LocalBackend

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestGetFreePort:
    """Tests for _get_free_port() method."""

    def test_returns_valid_port(self):
        """Test that _get_free_port returns a valid port number."""
        backend = LocalBackend()
        port = backend._get_free_port()

        # Port should be in valid range
        assert 1024 <= port <= 65535, f"Port {port} not in valid range 1024-65535"

    def test_port_is_actually_free(self):
        """Test that the returned port is actually free and can be bound."""
        backend = LocalBackend()
        port = backend._get_free_port()

        # Try to bind to the port to verify it's free
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                s.listen(1)
            except OSError as e:
                pytest.fail(f"Port {port} is not actually free: {e}")

    def test_multiple_calls_return_different_ports(self):
        """Test that multiple calls return different free ports."""
        backend = LocalBackend()

        ports = [backend._get_free_port() for _ in range(5)]

        # All ports should be unique
        assert len(ports) == len(set(ports)), "Some ports were duplicated"


class TestInitProcessGroupForWeightUpdates:
    """Tests for _init_process_group_for_weight_updates() method."""

    @pytest.mark.asyncio
    async def test_returns_correct_structure(self):
        """Test that method returns (init_method, world_size) tuple."""
        backend = LocalBackend()

        model = art.TrainableModel(
            name="test-model",
            project="test",
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            trainable=True,
        )

        init_method, world_size = await backend._init_process_group_for_weight_updates(
            model=model,
            num_actor_gpus=2,
        )

        # Check return types
        assert isinstance(init_method, str), "init_method should be a string"
        assert isinstance(world_size, int), "world_size should be an int"

    @pytest.mark.asyncio
    async def test_init_method_format(self):
        """Test that init_method has correct TCP format."""
        backend = LocalBackend()

        model = art.TrainableModel(
            name="test-model",
            project="test",
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            trainable=True,
        )

        init_method, _ = await backend._init_process_group_for_weight_updates(
            model=model,
            num_actor_gpus=1,
        )

        # Should start with tcp://localhost:
        assert init_method.startswith("tcp://localhost:"), (
            f"init_method should start with 'tcp://localhost:', got {init_method}"
        )

        # Extract port and verify it's valid
        port_str = init_method.split(":")[-1]
        port = int(port_str)
        assert 1024 <= port <= 65535, f"Port {port} not in valid range"

    @pytest.mark.asyncio
    async def test_world_size_calculation(self):
        """Test that world_size = 1 (trainer) + num_actor_gpus."""
        backend = LocalBackend()

        model = art.TrainableModel(
            name="test-model",
            project="test",
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            trainable=True,
        )

        test_cases = [
            (1, 2),  # 1 actor GPU -> world_size 2
            (2, 3),  # 2 actor GPUs -> world_size 3
            (4, 5),  # 4 actor GPUs -> world_size 5
        ]

        for num_actor_gpus, expected_world_size in test_cases:
            _, world_size = await backend._init_process_group_for_weight_updates(
                model=model,
                num_actor_gpus=num_actor_gpus,
            )
            assert world_size == expected_world_size, (
                f"For {num_actor_gpus} actor GPUs, expected world_size {expected_world_size}, got {world_size}"
            )

    @pytest.mark.asyncio
    async def test_multiple_calls_different_ports(self):
        """Test that multiple calls generate different init_methods (different ports)."""
        backend = LocalBackend()

        model = art.TrainableModel(
            name="test-model",
            project="test",
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            trainable=True,
        )

        init_methods = []
        for _ in range(3):
            init_method, _ = await backend._init_process_group_for_weight_updates(
                model=model,
                num_actor_gpus=1,
            )
            init_methods.append(init_method)

        # All init_methods should be unique (different ports)
        assert len(init_methods) == len(set(init_methods)), (
            "Multiple calls should generate different init_methods"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU for NCCL")
class TestJoinProcessGroupAsTrainer:
    """
    Tests for _join_process_group_as_trainer() method.

    NOTE: These tests require actual GPU hardware since they test real NCCL
    process group initialization. They will be skipped if no GPU is available.
    """

    @pytest.mark.asyncio
    async def test_basic_process_group_creation(self):
        """
        Test that process group can be created (with mock vLLM joining).

        This test simulates the full initialization sequence where:
        1. Trainer gets init_method and world_size
        2. A separate thread (simulating vLLM) joins the group
        3. Trainer joins the group
        4. Both are synchronized via NCCL rendezvous

        IMPORTANT: We use threading (not asyncio) because init_extra_process_group
        is a blocking/synchronous call. In real PipelineRL, vLLM runs in a separate
        process. Using a thread simulates this multi-process behavior.
        """
        logger.info("=" * 80)
        logger.info("TEST: test_basic_process_group_creation")
        logger.info("=" * 80)

        logger.info("Creating LocalBackend...")
        backend = LocalBackend()

        logger.info("Creating TrainableModel...")
        model = art.TrainableModel(
            name="test-model-pg",
            project="test",
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            trainable=True,
        )

        # Step 1: Get init_method
        logger.info("Step 1: Getting init_method and world_size...")
        init_method, world_size = await backend._init_process_group_for_weight_updates(
            model=model,
            num_actor_gpus=1,  # Single GPU for simplicity
        )
        logger.info(f"Got init_method: {init_method}, world_size: {world_size}")

        # Step 2: Simulate vLLM joining in a separate thread
        # IMPORTANT: We use a thread (not asyncio task) because init_extra_process_group
        # is a BLOCKING call, not async. In real PipelineRL, vLLM runs in a separate process.
        vllm_pg_result = [None]  # Use list to share result between threads
        vllm_error = [None]  # Capture any errors from vLLM thread

        def mock_vllm_join():
            """Simulate vLLM worker joining as rank 1 (runs in separate thread)."""
            try:
                logger.info("mock_vllm_join: Starting in separate thread...")
                import time

                from art.local.torch_utils import init_extra_process_group

                logger.info(
                    "mock_vllm_join: Sleeping for 0.5s to let trainer prepare..."
                )
                time.sleep(0.5)  # Give trainer time to start joining

                logger.info(
                    "mock_vllm_join: Calling init_extra_process_group as rank 1..."
                )
                logger.info(f"mock_vllm_join:   init_method={init_method}")
                logger.info(f"mock_vllm_join:   world_size={world_size}")
                logger.info(f"mock_vllm_join:   rank=1")

                vllm_pg = init_extra_process_group(
                    backend="nccl",
                    init_method=init_method,
                    rank=1,
                    world_size=world_size,
                    group_name="actor_vllm",
                )
                logger.info("mock_vllm_join: Successfully joined process group!")
                vllm_pg_result[0] = vllm_pg
            except Exception as e:
                logger.error(f"mock_vllm_join: ERROR: {type(e).__name__}: {e}")
                vllm_error[0] = e

        # Start mock vLLM join in background thread
        logger.info("Step 2: Starting mock vLLM join in background thread...")
        vllm_thread = threading.Thread(target=mock_vllm_join, daemon=True)
        vllm_thread.start()

        # Step 3: Trainer joins as rank 0
        logger.info("Step 3: Trainer joining process group as rank 0...")
        logger.info("Step 3: This will BLOCK until vLLM joins...")
        trainer_pg = await backend._join_process_group_as_trainer(
            init_method=init_method,
            world_size=world_size,
        )
        logger.info("Step 3: Trainer successfully joined!")

        # Wait for vLLM thread to finish
        logger.info("Step 4: Waiting for vLLM thread to complete...")
        vllm_thread.join(timeout=10.0)

        if vllm_thread.is_alive():
            logger.error("Step 4: vLLM thread is still alive after 10s!")
            pytest.fail("vLLM thread did not complete")

        if vllm_error[0]:
            logger.error(f"Step 4: vLLM thread had error: {vllm_error[0]}")
            raise vllm_error[0]

        vllm_pg = vllm_pg_result[0]
        logger.info("Step 4: vLLM thread completed!")

        # Step 5: Verify both process groups are valid
        logger.info("Step 5: Verifying process groups...")
        assert trainer_pg is not None, "Trainer process group should not be None"
        assert vllm_pg is not None, "vLLM process group should not be None"
        logger.info("  ✓ Both process groups are not None")

        # Both should have same world_size
        trainer_world_size = torch.distributed.get_world_size(group=trainer_pg)
        vllm_world_size = torch.distributed.get_world_size(group=vllm_pg)
        logger.info(f"  Trainer world_size: {trainer_world_size}")
        logger.info(f"  vLLM world_size: {vllm_world_size}")
        assert trainer_world_size == world_size, (
            f"Expected {world_size}, got {trainer_world_size}"
        )
        assert vllm_world_size == world_size, (
            f"Expected {world_size}, got {vllm_world_size}"
        )
        logger.info(f"  ✓ Both have correct world_size: {world_size}")

        # NOTE: We can't easily check ranks for extra/custom process groups
        # PyTorch's get_rank() requires a default process group to be initialized,
        # but we're creating extra groups (not the default). The important verification
        # is that both processes synchronized successfully via NCCL, which they did!
        logger.info("  ✓ Skipping rank check (limitation of extra process groups)")
        logger.info("  ✓ Both processes synchronized successfully via NCCL rendezvous")

        logger.info("=" * 80)
        logger.info("✓ Process group initialization successful!")
        logger.info("=" * 80)
        logger.info("✓ Process group initialization successful!")

        # Step 6: Cleanup - destroy process groups to allow test to complete
        logger.info("Step 6: Cleaning up process groups...")
        try:
            torch.distributed.destroy_process_group(trainer_pg)
            logger.info("  ✓ Trainer process group destroyed")
        except Exception as e:
            logger.warning(f"  Failed to destroy trainer process group: {e}")

        try:
            torch.distributed.destroy_process_group(vllm_pg)
            logger.info("  ✓ vLLM process group destroyed")
        except Exception as e:
            logger.warning(f"  Failed to destroy vLLM process group: {e}")

        logger.info("✓ Cleanup complete!")
        logger.info("=" * 80)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
