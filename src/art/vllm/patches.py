"""Monkey patches and modifications for vLLM."""

from typing import Any


def subclass_chat_completion_request() -> None:
    """
    Subclass ChatCompletionRequest so that logprobs are always returned.
    """
    import vllm.entrypoints.openai.protocol

    class ChatCompletionRequest(vllm.entrypoints.openai.protocol.ChatCompletionRequest):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.logprobs = True
            if self.top_logprobs is None:
                self.top_logprobs = 0

    vllm.entrypoints.openai.protocol.ChatCompletionRequest = ChatCompletionRequest


def patch_load_lora_adapter() -> None:
    """
    Allow reloading a LoRA adapter with the same name at runtime.

    Patch target:
      - vllm.entrypoints.openai.serving_models.OpenAIServingModels.load_lora_adapter
    """
    from vllm.entrypoints.openai.serving_models import (
        HTTPStatus,
        LoadLoRAAdapterRequest,
        LoRARequest,
        OpenAIServingModels,
        create_error_response,
        logger,
    )

    async def _patched_load_lora_adapter(
        self: OpenAIServingModels,
        request: LoadLoRAAdapterRequest,
        base_model_name: str | None = None,
    ):
        lora_name = request.lora_name
        lora_path = request.lora_path

        if not lora_name or not lora_path:
            return create_error_response(
                message="Both 'lora_name' and 'lora_path' must be provided.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        # Ensure atomicity based on the lora name
        async with self.lora_resolver_lock[lora_name]:
            if lora_name in self.lora_requests:
                lora_request = self.lora_requests[lora_name]
                lora_request.lora_path = lora_path
            else:
                unique_id = self.lora_id_counter.inc(1)
                lora_request = LoRARequest(
                    lora_name=lora_name,
                    lora_int_id=unique_id,
                    lora_path=lora_path,
                )

            if base_model_name is not None and self.is_base_model(base_model_name):
                lora_request.base_model_name = base_model_name

            # Validate that the adapter can be loaded into the engine.
            try:
                await self.engine_client.add_lora(lora_request)
            except Exception as e:
                error_type = "BadRequestError"
                status_code = HTTPStatus.BAD_REQUEST
                if "No adapter found" in str(e):
                    error_type = "NotFoundError"
                    status_code = HTTPStatus.NOT_FOUND

                return create_error_response(
                    message=str(e), err_type=error_type, status_code=status_code
                )

            self.lora_requests[lora_name] = lora_request
            logger.info(
                "Loaded LoRA adapter: name '%s', path '%s'", lora_name, lora_path
            )
            return f"Success: LoRA adapter '{lora_name}' added successfully."

    OpenAIServingModels.load_lora_adapter = _patched_load_lora_adapter


def patch_lora_cache_manager() -> None:
    """
    Allow in-place LoRA refreshes while avoiding per-request reloads.

    Patch targets:
      - vllm.lora.worker_manager.LRUCacheWorkerLoRAManager._apply_adapters
      - vllm.lora.worker_manager.LRUCacheWorkerLoRAManager.add_adapter
    """
    from vllm.lora.worker_manager import (
        LoRARequest,
        LRUCacheLoRAModelManager,
        LRUCacheWorkerLoRAManager,
    )

    def _patched__apply_adapters(
        self: LRUCacheWorkerLoRAManager, lora_requests: set[LoRARequest]
    ) -> None:
        loras_map = {
            lora_request.lora_int_id: lora_request
            for lora_request in lora_requests
            if lora_request
        }
        if len(loras_map) > self._adapter_manager.lora_slots:
            raise RuntimeError(
                f"Number of requested LoRAs ({len(loras_map)}) is greater "
                "than the number of GPU LoRA slots "
                f"({self._adapter_manager.lora_slots})."
            )
        for lora in loras_map.values():
            self.add_adapter(lora, force_load=False)

    def _patched_add_adapter(
        self: LRUCacheWorkerLoRAManager,
        lora_request: LoRARequest,
        force_load: bool = True,
    ) -> bool:
        # Note that this method is not thread-safe. It may be invoked multiple
        # times for the same adapter when using multiple API servers.
        # This is ok because it's currently only called from
        # the single-threaded core engine loop.

        if lora_request.lora_int_id not in self.list_adapters() or force_load:
            # Load the new adapter first to ensure it is actually valid, before
            # evicting any existing adapters.
            # This may cause the # of loaded lora adapters to very temporarily
            # exceed `--max-cpu-loras`.
            lora = self._load_adapter(lora_request)
            self._adapter_manager.remove_adapter(lora.id)

            # Loading succeeded, now check if we will exceed cache capacity and
            # evict if the oldest adapter if so
            if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                assert isinstance(self._adapter_manager, LRUCacheLoRAModelManager)
                self._adapter_manager.remove_oldest_adapter()
            # Then add the new adapter to the cache
            loaded = self._adapter_manager.add_adapter(lora)
        else:
            # If the lora is already loaded, just touch it to
            # update its position in the caches
            loaded = (
                self._adapter_manager.get_adapter(lora_request.lora_int_id) is not None
            )
        self._adapter_manager.activate_adapter(lora_request.lora_int_id)
        return loaded

    LRUCacheWorkerLoRAManager._apply_adapters = _patched__apply_adapters
    LRUCacheWorkerLoRAManager.add_adapter = _patched_add_adapter


def patch_lora_runtime_reload() -> None:
    patch_load_lora_adapter()
    patch_lora_cache_manager()


def patch_listen_for_disconnect() -> None:
    async def patched_listen_for_disconnect(request):
        try:
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    break
        except UnboundLocalError:
            pass

    # Replace the original function
    import vllm.entrypoints.utils

    vllm.entrypoints.utils.listen_for_disconnect = patched_listen_for_disconnect


def patch_tool_parser_manager() -> None:
    """
    Patch ToolParserManager to support streaming tool call logprobs.
    """
    from vllm.entrypoints.openai.protocol import DeltaMessage
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    get_tool_parser = ToolParserManager.get_tool_parser

    def patched_get_tool_parser(name: str) -> type:
        tool_parser_class = get_tool_parser(name)
        original = tool_parser_class.extract_tool_calls_streaming

        def patch(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            return original(*args, **kwargs) or DeltaMessage()

        tool_parser_class.extract_tool_calls_streaming = patch
        return tool_parser_class

    ToolParserManager.get_tool_parser = patched_get_tool_parser
