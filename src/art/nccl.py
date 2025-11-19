import logging
import pickle
from typing import Generator

import torch
from torch.distributed.tensor import DTensor
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

logger = logging.getLogger(__name__)


def stateless_init_process_group(
    master_address, master_port, rank, world_size, device, timeout
):
    pg = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=rank,
        world_size=world_size,
        store_timeout=timeout,
    )
    return PyNcclCommunicator(pg, device=device)


def send_state_dict(
    state_dict: dict[str, torch.Tensor], pg: PyNcclCommunicator
) -> None:
    """
    Get a state dict of tensor and broadcast it to the other ranks using NCCL.
    """
    # Group tensors by dtype
    dtype_groups: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    for key, value in state_dict.items():
        assert not isinstance(value, DTensor), (
            "DTensor is not supported for broadcast, should have been converted to tensor already"
        )
        dtype = value.dtype
        if dtype not in dtype_groups:
            dtype_groups[dtype] = []
        dtype_groups[dtype].append((key, value))

    # Build metadata: for each dtype group, store keys and shapes
    metadata = {}
    for dtype, items in dtype_groups.items():
        metadata[dtype] = [(key, value.shape, value.numel()) for key, value in items]

    # Send metadata
    state = pickle.dumps(metadata)
    size_tensor = torch.tensor([len(state)], dtype=torch.long).cuda()
    pg.broadcast(size_tensor, src=0)
    state_tensor = torch.ByteTensor(list(state)).cuda()
    pg.broadcast(state_tensor, src=0)

    # Concatenate and broadcast tensors grouped by dtype
    for dtype, items in dtype_groups.items():
        # Flatten all tensors and concatenate
        flat_tensors = [value.flatten() for _, value in items]
        concatenated = torch.cat(flat_tensors)
        pg.broadcast(concatenated, src=0)
        del concatenated
        # Clean up individual tensors
        for _, value in items:
            del value


def receive_state_dict(
    pg: PyNcclCommunicator,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Stream tensors in a state dict broadcasted over NCCL."""
    size_tensor = torch.tensor([10], dtype=torch.long).to(pg.device)
    pg.broadcast(size_tensor, src=0)
    state_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8).to(pg.device)
    pg.broadcast(state_tensor, src=0)

    metadata = pickle.loads(bytes(state_tensor.cpu().numpy()))

    # Receive concatenated tensors per dtype and split them back
    for dtype, tensor_info_list in metadata.items():
        # Receive concatenated tensor for this dtype
        total_elements = sum(numel for _, _, numel in tensor_info_list)
        concatenated = torch.empty(total_elements, dtype=dtype, device=pg.device)
        pg.broadcast(concatenated, src=0)

        # Split concatenated tensor back into individual tensors
        offset = 0
        for key, shape, numel in tensor_info_list:
            tensor = concatenated[offset : offset + numel].view(shape).clone()
            offset += numel
            try:
                yield key, tensor
            finally:
                del tensor

        del concatenated


def send_integer(integer: int, communicator: PyNcclCommunicator) -> None:
    """
    Send an integer to the other ranks using NCCL.
    """
    integer_tensor = torch.tensor([integer], dtype=torch.long).cuda()
    communicator.broadcast(integer_tensor, src=0)


def receive_integer(communicator: PyNcclCommunicator) -> int:
    """
    Receive an integer from the other ranks using NCCL.
    """
    integer_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    communicator.broadcast(integer_tensor, src=0)
    return integer_tensor.item()


def filter_state_dict_by_layers(
    state_dict: dict[str, torch.Tensor], num_layers: int
) -> Generator[tuple[int, dict[str, torch.Tensor]], None, None]:
    """
    Yield a generator of state dicts for each layer as well as the remaining weights.
    """

    yield (
        0,
        {key: value for key, value in state_dict.items() if "model.layers" not in key},
    )

    for i in range(1, num_layers + 1):  # +1 because layer indices start from 1
        yield (
            i,
            {
                key: value
                for key, value in state_dict.items()
                if key.startswith(f"model.layers.{i}.") or key == f"model.layers.{i}"
            },
        )


def get_max_layer_num(state_dict: dict[str, torch.Tensor]) -> int:
    """Get the maximum number of layers in the model."""
    return (
        max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1
    )


class NCCLBroadcastSender:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device,
        logger,
        timeout: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.logger = logger

        self.training_rank = 0

        if self.training_rank == 0:
            self.pg = stateless_init_process_group(
                host, port, rank, world_size, device, timeout
            )
            self.logger.info(
                f"NCCL broadcast initialized for rank {rank} and world size {world_size}"
            )

        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def broadcast_state_dict(self, model: torch.nn.Module) -> None:
        self.logger.info("Broadcasting weights to inference pool")

        state_dict = model.state_dict()
        self.logger.debug(f"State dict keys: {state_dict.keys()}")
        #
        num_layers = get_max_layer_num(state_dict)
        self.logger.info(f"Number of layers: {num_layers}")

        num_state_dict_to_send = (
            num_layers + 1
        )  # we send all layer plus the remaining weights

        if self.training_rank == 0:
            send_integer(num_state_dict_to_send, self.pg)

        self.logger.info(f"Broadcasting {num_state_dict_to_send} layer state dicts")

        for i, state_dict in filter_state_dict_by_layers(state_dict, num_layers):
            self.logger.debug(f"Sending layer {i}/{num_state_dict_to_send} state dict")
            for key, value in list(state_dict.items()):
                if isinstance(value, DTensor):
                    value = value.to(self.dtype).full_tensor()

                state_dict[key] = value

            if self.training_rank == 0:
                send_state_dict(state_dict, self.pg)

        self.logger.info("Weights broadcasted to inference pool")

    @torch.no_grad()
    def broadcast_lora(
        self, model: torch.nn.Module, peft_config: dict | None = None
    ) -> None:
        self.logger.debug("Broadcasting LoRA weights to inference pool")

        lora_tensors = {}
        # Extract LoRA tensors from state dict
        for key, value in model.state_dict().items():
            if "lora" in key:
                if isinstance(value, DTensor):
                    value = value.to(self.dtype).full_tensor()

                # Remove '.default' from the key if present, as vLLM doesn't expect it
                # e.g. base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
                # -> base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
                clean_key = key.replace(".default", "")
                lora_tensors[clean_key] = value

        self.logger.info(f"Found {len(lora_tensors)} LoRA tensors to broadcast")

        if self.training_rank == 0:
            # Send config if provided
            if peft_config is not None:
                config_bytes = pickle.dumps(peft_config)
                size_tensor = torch.tensor([len(config_bytes)], dtype=torch.long).cuda()
                self.pg.broadcast(size_tensor, src=0)
                config_tensor = torch.tensor(
                    list(config_bytes), dtype=torch.uint8
                ).cuda()
                self.pg.broadcast(config_tensor, src=0)
            else:
                # Send 0 size to indicate no config
                size_tensor = torch.tensor([0], dtype=torch.long).cuda()
                self.pg.broadcast(size_tensor, src=0)

            # Send 1 as number of chunks (we send all at once since LoRA is small)
            send_integer(1, self.pg)
            send_state_dict(lora_tensors, self.pg)

        self.logger.info("LoRA weights broadcasted to inference pool")


class NCCLBroadcastReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device,
        logger,
        timeout: int,
    ):
        self.logger = logger

        self.logger.info(
            f"Initializing NCCL broadcast receiver ({host}:{port}, rank={rank}, world_size={world_size})"
        )
        self.pg = stateless_init_process_group(
            host, port, rank, world_size, device, timeout
        )

        self.device = self.pg.device

    @torch.no_grad()
    def receive_state_dict(self):
        num_state_dict_to_receive = receive_integer(self.pg)

        self.logger.info(f"Receiving {num_state_dict_to_receive} state dicts")
        for i in range(num_state_dict_to_receive):
            self.logger.info(
                f"Receiving state dict {i}/{num_state_dict_to_receive}, peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
            )
            for key, value in receive_state_dict(self.pg):
                yield key, value

    @torch.no_grad()
    def receive_lora_dict(self) -> tuple[dict[str, torch.Tensor], dict | None]:
        """Receive all LoRA tensors and return as a dictionary, along with optional config."""

        # Receive config first
        size_tensor = torch.tensor([0], dtype=torch.long).to(self.device)
        self.pg.broadcast(size_tensor, src=0)
        config_size = size_tensor.item()

        peft_config = None
        if config_size > 0:
            config_tensor = torch.empty(config_size, dtype=torch.uint8).to(self.device)
            self.pg.broadcast(config_tensor, src=0)
            peft_config = pickle.loads(bytes(config_tensor.cpu().numpy()))

        tensors = {}
        for key, value in self.receive_state_dict():
            tensors[key] = value

        return tensors, peft_config
