"""
Torch distributed utilities for PipelineRL.

This module provides utilities for creating extra process groups for weight updates,
copied from the PipelineRL reference implementation.
"""

from datetime import timedelta
from typing import Any, Optional, Union

from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_extra_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    """
    Initialize an extra process group independent of the main process group.

    This allows creating multiple process groups with different world sizes,
    which is necessary for PipelineRL's weight update protocol where the
    actor process group (trainer + vLLM GPUs) is separate from the main
    training process group.

    Args:
        backend: Backend to use ('nccl', 'gloo', etc.)
        init_method: URL specifying how to initialize the process group
                     (e.g., 'tcp://localhost:23456', 'file:///tmp/pg')
        timeout: Timeout for operations
        world_size: Total number of processes in this group
        rank: Rank of the current process
        store: Optional pre-created store
        group_name: Name for this process group (used for prefixing keys)
        pg_options: Backend-specific options

    Returns:
        ProcessGroup object

    Example:
        >>> # Trainer process (rank 0)
        >>> pg = init_extra_process_group(
        ...     backend="nccl",
        ...     init_method="tcp://localhost:12345",
        ...     rank=0,
        ...     world_size=2,
        ...     group_name="actor",
        ... )

        >>> # vLLM process (rank 1)
        >>> pg = init_extra_process_group(
        ...     backend="nccl",
        ...     init_method="tcp://localhost:12345",
        ...     rank=1,
        ...     world_size=2,
        ...     group_name="actor",
        ... )
    """
    assert (store is None) or (
        init_method is None
    ), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        backend_options=pg_options,
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg
