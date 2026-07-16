from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.unified_cache_components import ComponentType
    from sglang.srt.mem_cache.unified_cache_components.tree_component import (
        TreeComponent,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class SWARecomputeConfig:
    """Model-specific sizing parameters for SWA-window recompute."""

    window_size: int
    gate_threshold: int

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.gate_threshold < self.window_size:
            raise ValueError("gate_threshold must be at least window_size")

    @classmethod
    def from_dimensions(
        cls,
        *,
        sliding_window_size: int,
        num_swa_layers: int,
        page_size: int,
        gate_multiplier: float,
    ) -> SWARecomputeConfig:
        if sliding_window_size <= 0:
            raise ValueError("sliding_window_size must be positive")
        if num_swa_layers <= 0:
            raise ValueError("num_swa_layers must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if not math.isfinite(gate_multiplier) or gate_multiplier < 1:
            raise ValueError("gate_multiplier must be finite and at least 1")

        # The last SWA layer needs the trailing W tokens, rounded up to a
        # multiple of page_size. Each preceding SWA layer extends the dependency
        # range backward by W tokens:
        # W_r = align_up(align_up(W, page_size) + (N - 1) * W, page_size).
        commit_tail = _ceil_div(sliding_window_size, page_size) * page_size
        unaligned_window = commit_tail + (num_swa_layers - 1) * sliding_window_size
        window_size = _ceil_div(unaligned_window, page_size) * page_size
        min_prefix_len = math.ceil(window_size * gate_multiplier)
        gate_threshold = _ceil_div(min_prefix_len, page_size) * page_size
        return cls(window_size=window_size, gate_threshold=gate_threshold)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclasses.dataclass
class CacheInitParams:
    disable: bool
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int

    is_eagle: bool = False
    tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    attn_cp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    attn_tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    pp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    eviction_policy: str = "lru"
    disable_finished_insert: bool = False

    enable_metrics: bool = False
    enable_kv_cache_events: bool = False
    enable_session_radix_cache: bool = False

    enable_mamba_extra_buffer: bool = False
    enable_mamba_extra_buffer_lazy: bool = False

    pp_rank: int = 0
    pp_size: int = 1

    attn_cp_rank: int = 0
    attn_cp_size: int = 1

    chunked_prefill_size: Optional[int] = None

    sliding_window_size: Optional[int] = None

    swa_recompute_config: Optional[SWARecomputeConfig] = None

    # Time-to-live for cache entries in seconds. If None, TTL is disabled.
    cache_ttl_seconds: Optional[float] = None

    tree_components: Optional[tuple[ComponentType, ...]] = None
    component_registry_override: Optional[dict[ComponentType, type[TreeComponent]]] = (
        None
    )
