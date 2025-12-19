"""
Tests for PipelineRLService selection in LocalBackend._get_service().

These tests verify that the correct service is selected based on model configuration flags.
The priority order should be:
1. PipelineRLService (if _use_pipeline_rl=True)
2. TorchtuneService (if torchtune_args is set)
3. DecoupledUnslothService (if _decouple_vllm_and_unsloth=True)
4. UnslothService (default)
"""

import os

import pytest

import art
from art import dev
from art.local.backend import LocalBackend
from art.local.pipeline_rl_service import PipelineRLService
from art.torchtune.service import TorchtuneService
from art.unsloth.decoupled_service import DecoupledUnslothService
from art.unsloth.service import UnslothService


@pytest.fixture
def backend():
    """Create a LocalBackend instance for testing."""
    # Use a temporary directory for testing
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="test_backend_service_selection_")
    # Use in_process=True to avoid mp_actors proxying for easier testing
    backend = LocalBackend(path=tmpdir, in_process=True)
    yield backend
    # Cleanup
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_pipeline_rl_service_selected_when_flag_is_true(backend):
    """Test that PipelineRLService is selected when _use_pipeline_rl=True."""
    model = art.TrainableModel(
        project="test-project",
        name="test-pipeline-rl",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        trainable=True,
        _internal_config={
            "_use_pipeline_rl": True,
            "init_args": {"max_seq_length": 2048},
            "peft_args": {"r": 16, "lora_alpha": 16},
            "trainer_args": {"output_dir": backend._path},
        },
    )

    await backend.register(model)

    service = await backend._get_service(model)

    assert isinstance(service, PipelineRLService), (
        f"Expected PipelineRLService but got {type(service)}"
    )


@pytest.mark.asyncio
async def test_unsloth_service_selected_when_flag_is_false(backend):
    """Test that UnslothService is selected when _use_pipeline_rl=False (default)."""
    model = art.TrainableModel(
        project="test-project",
        name="test-unsloth",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        trainable=True,
        _internal_config={
            "_use_pipeline_rl": False,
            "init_args": {"max_seq_length": 2048},
            "peft_args": {"r": 16, "lora_alpha": 16},
            "trainer_args": {"output_dir": backend._path},
        },
    )

    await backend.register(model)

    service = await backend._get_service(model)

    assert isinstance(service, UnslothService), (
        f"Expected UnslothService but got {type(service)}"
    )


@pytest.mark.asyncio
async def test_unsloth_service_selected_when_flag_not_set(backend):
    """Test that UnslothService is selected when _use_pipeline_rl is not set."""
    model = art.TrainableModel(
        project="test-project",
        name="test-unsloth-default",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        trainable=True,
        _internal_config={
            "init_args": {"max_seq_length": 2048},
            "peft_args": {"r": 16, "lora_alpha": 16},
            "trainer_args": {"output_dir": backend._path},
        },
    )

    await backend.register(model)

    service = await backend._get_service(model)

    assert isinstance(service, UnslothService), (
        f"Expected UnslothService but got {type(service)}"
    )


@pytest.mark.asyncio
async def test_decoupled_unsloth_service_selected_when_decouple_flag_is_true(backend):
    """Test that DecoupledUnslothService is selected when _decouple_vllm_and_unsloth=True."""
    model = art.TrainableModel(
        project="test-project",
        name="test-decoupled",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        trainable=True,
        _internal_config={
            "_decouple_vllm_and_unsloth": True,
            "init_args": {"max_seq_length": 2048},
            "peft_args": {"r": 16, "lora_alpha": 16},
            "trainer_args": {"output_dir": backend._path},
        },
    )

    await backend.register(model)

    service = await backend._get_service(model)

    assert isinstance(service, DecoupledUnslothService), (
        f"Expected DecoupledUnslothService but got {type(service)}"
    )


@pytest.mark.asyncio
async def test_pipeline_rl_has_priority_over_torchtune(backend):
    """Test that _use_pipeline_rl takes priority over torchtune_args."""
    model = art.TrainableModel(
        project="test-project",
        name="test-priority-torchtune",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        trainable=True,
        _internal_config={
            "_use_pipeline_rl": True,
            "torchtune_args": {"some": "config"},  # This should be ignored
            "init_args": {"max_seq_length": 2048},
            "peft_args": {"r": 16, "lora_alpha": 16},
            "trainer_args": {"output_dir": backend._path},
        },
    )

    await backend.register(model)

    service = await backend._get_service(model)

    assert isinstance(service, PipelineRLService), (
        f"Expected PipelineRLService (priority over TorchtuneService) but got {type(service)}"
    )


@pytest.mark.asyncio
async def test_pipeline_rl_has_priority_over_decoupled(backend):
    """Test that _use_pipeline_rl takes priority over _decouple_vllm_and_unsloth."""
    model = art.TrainableModel(
        project="test-project",
        name="test-priority-decoupled",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        trainable=True,
        _internal_config={
            "_use_pipeline_rl": True,
            "_decouple_vllm_and_unsloth": True,  # This should be ignored
            "init_args": {"max_seq_length": 2048},
            "peft_args": {"r": 16, "lora_alpha": 16},
            "trainer_args": {"output_dir": backend._path},
        },
    )

    await backend.register(model)

    service = await backend._get_service(model)

    assert isinstance(service, PipelineRLService), (
        f"Expected PipelineRLService (priority over DecoupledUnslothService) but got {type(service)}"
    )


@pytest.mark.asyncio
async def test_service_caching(backend):
    """Test that services are cached and reused for the same model."""
    model = art.TrainableModel(
        project="test-project",
        name="test-caching",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        trainable=True,
        _internal_config={
            "_use_pipeline_rl": True,
            "init_args": {"max_seq_length": 2048},
            "peft_args": {"r": 16, "lora_alpha": 16},
            "trainer_args": {"output_dir": backend._path},
        },
    )

    await backend.register(model)

    service1 = await backend._get_service(model)
    service2 = await backend._get_service(model)

    # Should be the exact same object (cached)
    assert service1 is service2, "Services should be cached and reused"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
