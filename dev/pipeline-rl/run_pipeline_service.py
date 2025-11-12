"""
PipelineRL Service Test Script

This script tests the PipelineRL service initialization:
- Trainer on GPU 0
- vLLM inference server on GPU 1
- Service initialization and vLLM startup

Usage:
    CUDA_VISIBLE_DEVICES=0,1 uv run python dev/pipeline-rl/run_pipeline_service.py
"""

import asyncio
import logging
import os

import art
from art.dev.model import InternalModelConfig
from art.local.backend import LocalBackend


class DeviceContextFilter(logging.Filter):
    """Filter that adds device and rank information to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        # RANK/LOCAL_RANK are set by torchrun; fall back to 'NA'
        record.rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "NA"))

        # Choose device
        try:
            import torch

            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                record.device = f"cuda:{current_device}({device_name[:20]}...)"
            else:
                record.device = "cpu"
        except Exception:
            # Fallback to CUDA_VISIBLE_DEVICES if torch not available
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
            record.device = f"CUDA_VIS:{cuda_visible}"

        return True


# Configure logging to show all output
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

if not root_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format with PID, rank, device info
    fmt = "%(asctime)s | pid=%(process)d | rank=%(rank)s | dev=%(device)s | %(name)s | %(levelname)s | %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
    console_handler.setFormatter(formatter)

    # Add filter to inject device/rank info
    console_handler.addFilter(DeviceContextFilter())

    root_logger.addHandler(console_handler)

# Explicitly set levels for ART loggers
logging.getLogger("art.local.backend").setLevel(logging.INFO)
logging.getLogger("art.local.pipeline_rl_service").setLevel(logging.INFO)
logging.getLogger("art.unsloth").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


async def main():
    logger.info("=" * 80)
    logger.info("PipelineRL Service Test")
    logger.info("=" * 80)
    logger.info("")

    # Show current CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
    logger.info(f"Current CUDA_VISIBLE_DEVICES: {cuda_visible}")
    logger.info("")

    # Declare the model
    logger.info("[1] Creating TrainableModel with PipelineRL configuration...")
    model = art.TrainableModel(
        name="agent-001",
        project="pipeline-rl-test",
        base_model="OpenPipe/Qwen3-14B-Instruct",
        _internal_config=InternalModelConfig(
            _use_pipeline_rl=True,
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
    logger.info("      - Create PipelineRLService")
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
    asyncio.run(main())
