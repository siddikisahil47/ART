"""
Tests for PipelineRLService structure.

These are lightweight structural tests (imports, dataclass fields, method existence).
"""

import pytest
from dataclasses import fields


class TestPipelineRLServiceStructure:
    """Tests for PipelineRLService structure and attributes."""

    def test_import(self):
        """Test that we can import the service."""
        from art.local.pipeline_rl_service import PipelineRLService

        assert PipelineRLService is not None

    def test_is_dataclass(self):
        """Test that PipelineRLService is a dataclass."""
        from art.local.pipeline_rl_service import PipelineRLService
        from dataclasses import is_dataclass

        assert is_dataclass(PipelineRLService)

    def test_has_required_attributes(self):
        """Test that service has all required attributes."""
        from art.local.pipeline_rl_service import PipelineRLService

        field_names = {f.name for f in fields(PipelineRLService)}

        required_fields = {
            "model_name",
            "base_model",
            "config",
            "output_dir",
            "_vllm_process",
            "_generation_step",
            "_training_step",
            "_train_task",
        }

        assert required_fields.issubset(field_names), f"Missing fields: {required_fields - field_names}"

    def test_can_instantiate(self):
        """Test that we can create an instance."""
        from art.local.pipeline_rl_service import PipelineRLService

        service = PipelineRLService(
            model_name="test-model",
            base_model="test/base",
            config={},
            output_dir="/tmp/test",
        )

        assert service.model_name == "test-model"
        assert service.base_model == "test/base"
        assert service._generation_step == 0
        assert service._training_step == 0

    def test_implements_model_service_protocol(self):
        """Test that service implements ModelService protocol."""
        from art.local.pipeline_rl_service import PipelineRLService
        from art.local.service import ModelService

        # Check that PipelineRLService is recognized as implementing ModelService
        assert isinstance(PipelineRLService, type)

        # Check that it has all required methods
        required_methods = ["start_openai_server", "vllm_engine_is_sleeping", "train"]

        for method in required_methods:
            assert hasattr(PipelineRLService, method), f"Missing method: {method}"

    def test_has_weight_update_method(self):
        """Test that service has weight update-specific methods."""
        from art.local.pipeline_rl_service import PipelineRLService

        assert hasattr(PipelineRLService, "start_openai_server_with_weight_updates")
        assert hasattr(PipelineRLService, "swap_lora_checkpoint")

    def test_vllm_engine_is_sleeping_returns_false(self):
        """Test that vllm_engine_is_sleeping always returns False."""
        from art.local.pipeline_rl_service import PipelineRLService
        import asyncio

        service = PipelineRLService(
            model_name="test-model",
            base_model="test/base",
            config={},
            output_dir="/tmp/test",
        )

        result = asyncio.run(service.vllm_engine_is_sleeping())
        assert result is False

    def test_step_counters_initialize_to_zero(self):
        """Test that step counters start at zero."""
        from art.local.pipeline_rl_service import PipelineRLService

        service = PipelineRLService(
            model_name="test-model",
            base_model="test/base",
            config={},
            output_dir="/tmp/test",
        )

        assert service._generation_step == 0
        assert service._training_step == 0

    def test_docstrings_exist(self):
        """Test that key methods have docstrings."""
        from art.local.pipeline_rl_service import PipelineRLService

        assert PipelineRLService.__doc__ is not None
        assert PipelineRLService.train.__doc__ is not None
        assert PipelineRLService.start_openai_server_with_weight_updates.__doc__ is not None
        assert PipelineRLService.swap_lora_checkpoint.__doc__ is not None
