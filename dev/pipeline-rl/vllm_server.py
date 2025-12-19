"""
Minimal vLLM server with weight update capabilities.
Based on PipelineRL
"""

import logging
import os
import signal

# Import from ART's torch_utils (already copied to src/art/local/torch_utils.py)
import sys
import time
from pathlib import Path
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)

import torch
import uvloop
from pydantic import BaseModel, Field
from vllm._version import version
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    init_app_state,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, set_ulimit
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

# Add ART src to path so we can import art.local.torch_utils
art_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(art_src))
from art.local.torch_utils import init_extra_process_group

# ====== Environment Variables (from PipelineRL launch.py) ======
# Set critical environment variables before any torch operations
os.environ["NCCL_CUMEM_ENABLE"] = "0"  # Disable NCCL CUMEM for stability
os.environ["TORCH_DISABLE_SHARE_RDZV_TCP_STORE"] = "1"  # Disable shared RDZV TCP store
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"  # Cleaner logs
os.environ["VLLM_LOGGING_LEVEL"] = os.environ.get("VLLM_LOGGING_LEVEL", "INFO")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # Compatibility for weight loading

# With CUDA_VISIBLE_DEVICES=0 for vLLM and CUDA_VISIBLE_DEVICES=1 for trainer:
# - P2P: cudaDev IDs don't match physical devices, causing setup failures
# - SHM: shared memory segment coordination fails between processes
# Force pure network/socket transport which works reliably across processes
if "NCCL_P2P_DISABLE" not in os.environ:
    os.environ["NCCL_P2P_DISABLE"] = "1"
if "NCCL_SHM_DISABLE" not in os.environ:
    os.environ["NCCL_SHM_DISABLE"] = "1"

# ====== Logging Setup ======
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


# ====== Protocol and Data Models ======
@runtime_checkable
class LikeWorker(Protocol):
    """Protocol for worker interface used in weight updates"""

    rank: int
    local_rank: int
    device: torch.device
    model_runner: GPUModelRunner
    pg_rank: int
    process_group: Any
    model_config: ModelConfig


class ParameterInfo(BaseModel):
    """Information about a model parameter for weight updates"""

    name: str
    shape: List[int]
    dtype: str


class WeightUpdateRequest(BaseModel):
    """Request format for weight updates from trainer to vLLM"""

    kind: Literal["weight_update_request"] = "weight_update_request"
    version: int = 1
    parameters_info: List[ParameterInfo]
    timestamp: float = Field(default_factory=lambda: time.time())


# ====== Worker Extension for Weight Updates ======


class WorkerExtension:
    """
    Extension for vLLM workers to enable receiving weight updates from the trainer.
    Matches PipelineRL's implementation exactly.
    """

    def init_actor_update_group(
        self: LikeWorker,
        actor_idx: int,
        actor_ngpus: int,
        weight_update_group_init_method: str,
        weight_update_group_world_size: int,
    ):
        """Initialize the process group for receiving weight updates (matches PipelineRL)"""
        self.pg_rank = 1 + actor_idx * actor_ngpus + self.rank
        # Log all you know (matching PipelineRL's logging)
        prefix = "[INIT_ACTOR_UPDATE_GROUP]: "
        logger.info(
            prefix
            + f"Actor index: {actor_idx}, actor ngpus: {actor_ngpus}, rank: {self.rank}, pg_rank: {self.pg_rank}"
        )
        logger.info(
            prefix
            + f"Weight update group init method: {weight_update_group_init_method}, "
            f"world size: {weight_update_group_world_size}"
        )

        # Validation: Check if pg_rank is within world_size bounds
        if self.pg_rank >= weight_update_group_world_size:
            raise ValueError(
                f"Process group rank {self.pg_rank} is out of bounds for world_size {weight_update_group_world_size}. "
                f"This usually means actor_ngpus ({actor_ngpus}) is incorrect. "
                f"Expected actor_ngpus=1 for world_size=2 setup with 1 trainer + 1 vLLM GPU. "
                f"Check CUDA_VISIBLE_DEVICES - vLLM should only see 1 GPU!"
            )

        self.process_group = init_extra_process_group(
            group_name="actor",
            backend="nccl",
            init_method=weight_update_group_init_method,
            rank=self.pg_rank,
            world_size=weight_update_group_world_size,
        )

    def receive_weight_update(self: LikeWorker, request_dict: dict):
        # Reconstruct WeightUpdateRequest from dict
        request = WeightUpdateRequest(**request_dict)

        logger.info(
            f"[Worker {self.rank}] receive_weight_update called with request: {request}"
        )
        torch.cuda.synchronize(self.device)
        logger.info(f"[Worker {self.rank}] Start receiving weight update")
        for info in request.parameters_info:
            logger.info(f"[Worker {self.rank}] Update weight for {info.name}")
            model_dtype = self.model_config.dtype
            assert info.dtype == str(model_dtype), (
                f"mismatch dtype: src {info.dtype}, dst {self.model_config.dtype}"
            )
            logger.info(f"[Worker {self.rank}] - weight types checked")
            buffer = torch.empty(
                tuple(info.shape), dtype=model_dtype, device=self.device
            )
            logger.info(f"[Worker {self.rank}] - buffer created, about to broadcast...")
            logger.info(
                f"[Worker {self.rank}] - process_group: {self.process_group}, pg_rank: {self.pg_rank}"
            )
            torch.distributed.broadcast(buffer, src=0, group=self.process_group)
            logger.info(f"[Worker {self.rank}] - buffer broadcasted")
            loaded_params = self.model_runner.model.load_weights(
                weights=[(info.name, buffer)]
            )  # type: ignore
            logger.info("- weights loaded")
            if len(loaded_params) != 1:
                raise ValueError(f"model {info.name} not found in model state dict")
        logger.info("Weight update received")


# ====== Weight Update Manager ======


class WeightUpdateManager:
    """
    Manages weight updates between the trainer and vLLM engine.
    Matches PipelineRL's implementation with eager initialization.
    """

    def __init__(self, args, engine_client: AsyncMPClient):
        self.args = args
        self.engine_client = engine_client

    async def input_process_groups(self):
        """Initialize process groups eagerly during server startup (matches PipelineRL)"""
        await self.engine_client.collective_rpc_async(
            "init_actor_update_group",
            args=(
                self.args.actor_llm_idx,
                torch.cuda.device_count(),
                self.args.weight_update_group_init_method,
                self.args.weight_update_group_world_size,
            ),
        )

    async def receive_weight_update(self, request: WeightUpdateRequest):
        """Receive and propagate weight updates to workers (matches PipelineRL)"""
        # Send request directly through vLLM RPC
        logger.info(f"Weight update RPC called with: {request}")
        logger.info("About to call collective_rpc_async('receive_weight_update')...")

        # vLLM requires serializable types for RPC (or set VLLM_ALLOW_INSECURE_SERIALIZATION=1)
        # Convert Pydantic model to dict before sending
        request_dict = request.model_dump()
        logger.info(f"Converted request to dict: {request_dict}")

        try:
            result = await self.engine_client.collective_rpc_async(
                "receive_weight_update", args=(request_dict,)
            )
            logger.info(f"collective_rpc_async returned: {result}")
        except Exception as e:
            logger.error(f"collective_rpc_async failed: {e}", exc_info=True)
            raise
        logger.info("Weight update processed")


# ====== Helper Functions ======


def save_startup_script(args, script_path: Optional[Path] = None):
    """
    Save a startup script that can be used to restart the server with the same arguments.
    Similar to PipelineRL's save_command function.
    """
    if script_path is None:
        script_path = Path("./start_vllm_server.sh")

    # Reconstruct the command line
    cmd = [sys.executable, __file__]

    # Add all arguments
    for arg, value in vars(args).items():
        arg_name = arg.replace("_", "-")
        if value is True:
            cmd.append(f"--{arg_name}")
        elif value is not False and value is not None:
            cmd.append(f"--{arg_name}")
            cmd.append(str(value))

    # Write the script
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated startup script for vLLM server with weight updates\n")
        f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Add environment variables
        f.write("# Environment variables\n")
        f.write(
            f"export CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n"
        )
        f.write(f"export NCCL_CUMEM_ENABLE=0\n")
        f.write(f"export TORCH_DISABLE_SHARE_RDZV_TCP_STORE=1\n")
        f.write(f"export HF_DATASETS_DISABLE_PROGRESS_BARS=1\n")
        f.write(
            f"export VLLM_LOGGING_LEVEL={os.environ.get('VLLM_LOGGING_LEVEL', 'INFO')}\n"
        )
        f.write(f"export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1\n\n")

        # Add the command
        quoted_cmd = []
        for arg in cmd:
            if " " in arg or "$" in arg:
                quoted_cmd.append(f"'{arg}'")
            else:
                quoted_cmd.append(arg)
        f.write(" ".join(quoted_cmd) + "\n")

    # Make executable
    os.chmod(script_path, 0o755)
    logger.info(f"Saved startup script to {script_path}")


def check_model_path(model_path: str, finetune_checkpoint: Optional[str] = None) -> str:
    """
    Check and resolve the model path, similar to PipelineRL's handling.
    If a finetune checkpoint exists, use that instead of the base model.
    """
    # Check for finetune checkpoint first (like PipelineRL does)
    if finetune_checkpoint and os.path.exists(finetune_checkpoint):
        logger.info(f"Using finetuned model from: {finetune_checkpoint}")
        return finetune_checkpoint

    # Otherwise use the provided model path
    if os.path.exists(model_path):
        logger.info(f"Using local model from: {model_path}")
    else:
        logger.info(f"Using HuggingFace model: {model_path}")

    return model_path


# ====== Main Server Implementation ======


async def run_server(args, **uvicorn_kwargs) -> None:
    """
    Main server function that sets up and runs the vLLM server with weight update capabilities.
    """
    logger.info("vLLM API server version %s", version)
    logger.info("args: %s", args)

    # Tool parser setup
    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    valid_tool_parsers = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice and args.tool_call_parser not in valid_tool_parsers:
        raise KeyError(
            f"invalid tool call parser: {args.tool_call_parser} (choose from {{{','.join(valid_tool_parsers)}}})"
        )

    # Bind port before engine setup and reuse this socket for serving
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # Set safe keep-alive and ulimits
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    # Configure and create the vLLM engine
    engine_args = AsyncEngineArgs.from_cli_args(args)

    # Plug in our worker extension for weight updates
    # This needs to be the module path to THIS file's WorkerExtension class
    # TODO: make this a full path
    engine_args.worker_extension_cls = "__main__.WorkerExtension"

    engine_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    engine = AsyncLLM.from_vllm_config(
        vllm_config=engine_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        disable_log_stats=engine_args.disable_log_stats,
        disable_log_requests=engine_args.disable_log_requests,
    )
    assert isinstance(engine.engine_core, AsyncMPClient)

    # Initialize weight update manager (eager initialization, matching PipelineRL)
    weight_update_manager = WeightUpdateManager(args, engine.engine_core)
    if not args.disable_weight_updates:
        await weight_update_manager.input_process_groups()

    # Build FastAPI app and add weight update endpoint
    app = build_app(args)

    @app.post("/receive_weight_update")
    async def _receive_weight_update(request: WeightUpdateRequest):
        """Endpoint to receive weight updates from the trainer"""
        logger.info("received weight update request")
        await weight_update_manager.receive_weight_update(request)
        return {"status": "ok"}

    # Initialize app state and run HTTP server
    await init_app_state(engine, engine_config, app.state, args)
    shutdown_task = await serve_http(
        app,
        sock,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        timeout_keep_alive=60,  # larger timeout for weight updates
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )
    await shutdown_task
    sock.close()


def main():
    """
    Main entry point for the vLLM server with weight update support.
    This follows the same patterns as PipelineRL's launch.py run_actor_llm function.
    """
    parser = FlexibleArgumentParser(
        description="Patched vLLM OpenAI-Compatible server with weight updates for online RLHF."
    )
    # Add all standard vLLM arguments
    parser = make_arg_parser(parser)

    # Add custom arguments for weight updates (from PipelineRL)
    parser.add_argument(
        "--disable-weight-updates",
        action="store_true",
        help="Disable receiving weight updates from the trainer",
    )
    parser.add_argument(
        "--actor-llm-idx",
        type=int,
        default=0,
        help="Index of this actor LLM instance (for multi-actor setups)",
    )
    parser.add_argument(
        "--weight-update-group-init-method",
        type=str,
        required=False,  # Made optional with validation below
        help="TCP address for weight update group initialization (e.g., tcp://host:port)",
    )
    parser.add_argument(
        "--weight-update-group-world-size",
        type=int,
        required=False,  # Made optional with validation below
        help="Total world size for weight update group (trainer + all actor GPUs)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for log files",
    )
    parser.add_argument(
        "--finetune-checkpoint-path",
        type=str,
        default=None,
        help="Path to finetuned model checkpoint (overrides --model if exists)",
    )
    parser.add_argument(
        "--save-startup-script",
        type=str,
        default=None,
        help="Path to save a startup script for this server configuration",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without starting the server",
    )

    args = parser.parse_args()

    # Validate weight update arguments if weight updates are enabled
    if not args.disable_weight_updates:
        if not args.weight_update_group_init_method:
            parser.error(
                "--weight-update-group-init-method is required when weight updates are enabled"
            )
        if not args.weight_update_group_world_size:
            parser.error(
                "--weight-update-group-world-size is required when weight updates are enabled"
            )

    # Set up logging to file if specified (following PipelineRL pattern)
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler for logging
        file_handler = logging.FileHandler(log_dir / "vllm_server.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Redirect stdout/stderr to files (like PipelineRL does)
        sys.stdout = open(log_dir / "stdout.log", "a")
        sys.stderr = open(log_dir / "stderr.log", "a")

    # Check and potentially override model path with finetune checkpoint
    if hasattr(args, "model") and args.model:
        args.model = check_model_path(args.model, args.finetune_checkpoint_path)

    # Save startup script if requested
    if args.save_startup_script:
        save_startup_script(args, Path(args.save_startup_script))

    # Log startup info
    logger.info(f"Starting vLLM server with weight update support")
    logger.info(f"Model: {args.model if hasattr(args, 'model') else 'not specified'}")
    logger.info(f"Port: {args.port if hasattr(args, 'port') else 'default'}")
    logger.info(
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
    )
    logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    logger.info(
        f"Weight updates: {'DISABLED' if args.disable_weight_updates else 'ENABLED'}"
    )

    if not args.disable_weight_updates:
        logger.info(
            f"Weight update init method: {args.weight_update_group_init_method}"
        )
        logger.info(f"Weight update world size: {args.weight_update_group_world_size}")
        logger.info(f"Actor LLM index: {args.actor_llm_idx}")

    # Dry run mode - just print config and exit
    if args.dry_run or os.environ.get("DRY_RUN", "0") == "1":
        logger.info(
            "DRY RUN MODE - Configuration validated, exiting without starting server"
        )
        print("\n=== Configuration Summary ===")
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        print("\n=== Environment ===")
        print(f"  CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        return

    # Validate standard vLLM arguments
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
