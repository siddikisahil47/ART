import multiprocessing
import os
from argparse import Namespace
from contextlib import asynccontextmanager
from multiprocessing import forkserver
from typing import Any, AsyncIterator, Optional

import uvloop
from fastapi import Request

# Apply patches before importing vllm.entrypoints.openai modules
from art.vllm.patches import (
    patch_listen_for_disconnect,
    patch_lora_runtime_reload,
    patch_tool_parser_manager,
    subclass_chat_completion_request,
)

patch_listen_for_disconnect()
patch_lora_runtime_reload()
patch_tool_parser_manager()
# We must subclass ChatCompletionRequest before importing api_server
# or logprobs will not always be returned
subclass_chat_completion_request()

from vllm import AsyncEngineArgs, envs  # noqa: E402
from vllm.engine.protocol import EngineClient  # noqa: E402
from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server  # noqa: E402
from vllm.entrypoints.launcher import serve_http  # noqa: E402
from vllm.entrypoints.openai.api_server import (  # noqa: E402
    build_app,
    build_async_engine_client_from_engine_args,
    init_app_state,
    load_log_config,
    maybe_register_tokenizer_info_endpoint,
    setup_server,
)
from vllm.entrypoints.openai.cli_args import (  # noqa: E402
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.openai.tool_parsers import ToolParserManager  # noqa: E402
from vllm.logger import init_logger  # noqa: E402
from vllm.usage.usage_lib import UsageContext  # noqa: E402
from vllm.utils import FlexibleArgumentParser, decorate_logs  # noqa: E402

logger = init_logger("vllm.entrypoints.openai.api_server")


# copied from vllm/entrypoints/openai/api_server.py
async def run_custom_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server."""

    # Add process-specific prefix to stdout and stderr.
    decorate_logs("APIServer")

    listen_address, sock = setup_server(args)
    await run_custom_server_worker(listen_address, sock, args, **uvicorn_kwargs)


# copied from vllm/entrypoints/openai/api_server.py
async def run_custom_server_worker(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
    """Run a single API server worker."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    server_index = client_config.get("client_index", 0) if client_config else 0

    # Load logging config for uvicorn if specified
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with build_custom_async_engine_client(
        args,
        client_config=client_config,
    ) as engine_client:
        maybe_register_tokenizer_info_endpoint(args)
        app = build_app(args)

        @app.post("/init_broadcaster")
        async def _init_broadcaster(request: Request):
            data = await request.json()
            host = data.get("host")
            port = data.get("port")
            timeout = data.get("timeout")
            server_rank = data.get("server_rank")
            num_inference_server = data.get("num_inference_server")
            await engine_client.collective_rpc(
                "init_broadcaster",
                args=(host, port, server_rank, num_inference_server, timeout),
            )
            return {"status": "ok"}

        @app.post("/update_weights")
        async def _update_weights(request: Request):
            await engine_client.collective_rpc(
                "update_weights",
            )
            return {"status": "ok"}

        @app.post("/update_lora_weights")
        async def _update_lora_weights(request: Request):
            await engine_client.collective_rpc(
                "update_lora_weights",
            )
            return {"status": "ok"}

        vllm_config = await engine_client.get_vllm_config()
        await init_app_state(engine_client, vllm_config, app.state, args)

        logger.info("Starting vLLM API server %d on %s", server_index, listen_address)
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
            h11_max_header_count=args.h11_max_header_count,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


# copied from vllm/entrypoints/openai/api_server.py
@asynccontextmanager
async def build_custom_async_engine_client(
    args: Namespace,
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    disable_frontend_multiprocessing: Optional[bool] = None,
    client_config: Optional[dict[str, Any]] = None,
) -> AsyncIterator[EngineClient]:
    if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
        # The executor is expected to be mp.
        # Pre-import heavy modules in the forkserver process
        logger.debug("Setup forkserver with pre-imports")
        multiprocessing.set_start_method("forkserver")
        multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
        forkserver.ensure_running()
        logger.debug("Forkserver setup complete!")

    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    engine_args = AsyncEngineArgs.from_cli_args(args)
    # Add NCCLWeightUpdateWorker to engine
    engine_args.worker_extension_cls = "art.vllm.nccl_worker.NCCLWeightUpdateWorker"

    if disable_frontend_multiprocessing is None:
        disable_frontend_multiprocessing = bool(args.disable_frontend_multiprocessing)

    async with build_async_engine_client_from_engine_args(
        engine_args,
        usage_context=usage_context,
        disable_frontend_multiprocessing=disable_frontend_multiprocessing,
        client_config=client_config,
    ) as engine:
        yield engine


# copied from vllm/entrypoints/cli/serve.py
def server():
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    assert args is not None
    validate_parsed_serve_args(args)

    if hasattr(args, "model_tag") and args.model_tag is not None:
        args.model = args.model_tag

    if args.headless or args.api_server_count < 1:
        run_headless(args)
    else:
        if args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            # Single API server (this process).
            uvloop.run(run_custom_server(args))


if __name__ == "__main__":
    server()
