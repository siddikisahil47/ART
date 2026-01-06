from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.v1.worker.gpu_worker import Worker

from art.nccl import NCCLBroadcastReceiver

# This is to get type hints for the Worker class but not actually extend it at runtime as they conflict with each other:
# AssertionError: Worker class <class 'vllm.v1.worker.gpu_worker.Worker'> already has an attribute _eplb_after_scale_up,
# which conflicts with the worker extension class <class '... NCCLWeightUpdateWorker'>.

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class NCCLWeightUpdateWorker(Worker):
    """
    This is an vLLM worker extension for updating weights to an updated RL policy model using NCCL.
    """

    def init_broadcaster(
        self,
        host: str,
        port: int,
        server_rank: int,
        num_inference_server: int,
        timeout: int,
    ) -> None:
        """Initialize the process group for NCCL broadcast."""
        logger = init_logger("vllm.inference.vllm.worker_nccl")
        self.tp_rank = get_tp_group().rank

        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank

        global_rank_inference = (server_rank * tp_size) + tp_rank
        global_inference_world_size = num_inference_server * tp_size

        logger.info(
            f"Worker [tp={tp_rank} server_rank={server_rank}] -> [global_rank={global_rank_inference} global_world_size={global_inference_world_size}]"
        )
        logger.info(
            f"Model state dict keys: {self.model_runner.model.state_dict().keys()}"
        )

        self.nccl_broadcast = NCCLBroadcastReceiver(
            host=host,
            port=port,
            rank=global_rank_inference + 1,  # +1 as the trainer broadcaster is rank 0
            world_size=global_inference_world_size
            + 1,  # +1 for the trainer broadcaster
            device=self.device,
            logger=logger,
            timeout=timeout,
        )

    def update_weights(self) -> None:
        """Update weights with the nccl communicator."""
        model_runner = self.model_runner
        model = model_runner.model
        assert isinstance(model, Module)

        state_iter = self.nccl_broadcast.receive_state_dict()
        model.load_weights(state_iter)  # type: ignore

        # # Process weights after loading (important for some models)
        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)

    def update_lora_weights(self) -> None:
        """Update LoRA weights with the nccl communicator."""
        logger = init_logger("vllm.inference.vllm.worker_nccl")

        tensors, peft_config_dict = self.nccl_broadcast.receive_lora_dict()

        if peft_config_dict is None:
            # Fallback to existing config if not provided (risky if it changed)
            logger.warning(
                "No PEFT config received, using worker's default LoRA config"
            )
            # This might fail if lora_config is the global config, but we try our best
            peft_helper = PEFTHelper.from_dict(self.model_runner.lora_config.__dict__)
        else:
            peft_helper = PEFTHelper.from_dict(peft_config_dict)

        logger.info(f"[NCCL_WORKER] DEBUG: Tensors keys: {tensors.keys()}")

        lora_id = 1

        # Create LoRAModel from tensors
        lora_model = LoRAModel.from_lora_tensors(
            lora_model_id=lora_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device=self.device,
            dtype=list(tensors.values())[0].dtype if tensors else torch.bfloat16,
            # even though our LoRA weights are not applied on embeddings, we provide these because of the assertion in from_lora_tensors
            # https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/lora/models.py#L160
            # TODO: maybe this can be gotten from lora_manager.embeddings_modules or lora_manager.embedding_padding_modules
            embedding_modules={
                "embed_tokens": "input_embeddings",
                "lm_head": "output_embeddings",
            },
            embedding_padding_modules=["lm_head"],
        )

        # Update the adapter in the LoRA manager
        lora_manager = self.model_runner.lora_manager

        # TODO: how to actually apply LoRA weights into model_runner?
