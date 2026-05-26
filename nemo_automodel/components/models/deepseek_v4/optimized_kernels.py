# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional DeepSeek V4 optimized kernel dispatch.

The torch implementations below are kept as the numerical reference.  Optional
TileLang-backed paths are sourced from:

* Sinkhorn: imported from DeepSeek TileKernels
  ``tile_kernels.modeling.mhc.ops.sinkhorn_normalize``.  No TileKernels source
  is vendored in AutoModel.  Upstream source:
  https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/modeling/mhc/ops/sinkhorn.py
  Upstream license: MIT, copyright 2026 DeepSeek.
* Sparse attention and indexer: vendored/adapted Miles DeepSeek V4 ops in
  ``nemo_automodel.components.models.deepseek_v4.kernels``.  Upstream source:
  https://github.com/yueming-yuan/miles/tree/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops
  Upstream license: Apache-2.0, copyright 2025 Zhipu AI.  See
  ``nemo_automodel/components/models/deepseek_v4/kernels/__init__.py`` for
  the per-file attribution.

Those packages are imported with ``safe_import`` so environments without
TileLang still import the model and use the existing torch path.
"""

from __future__ import annotations

import os
from typing import Literal

import torch

from nemo_automodel.shared.import_utils import safe_import_from

Dsv4SparseAttentionBackend = Literal["torch", "sparse_torch", "tilelang", "auto"]
Dsv4IndexerBackend = Literal["torch", "tilelang", "auto"]
Dsv4SinkhornBackend = Literal["torch", "tilelang", "auto"]

_HAS_TILE_KERNELS_SINKHORN, _tile_kernels_sinkhorn = safe_import_from(
    "tile_kernels.modeling.mhc.ops",
    "sinkhorn_normalize",
    msg="TileKernels sinkhorn is unavailable. Install tile_kernels and tilelang to use backend.attn='tilelang'.",
)
_HAS_TILE_KERNELS_SINKHORN_FWD, _tile_kernels_sinkhorn_fwd = safe_import_from(
    "tile_kernels.mhc.sinkhorn_kernel",
    "_mhc_sinkhorn_fwd",
    msg="TileKernels low-level sinkhorn forward kernel is unavailable.",
)
_HAS_TILE_KERNELS_SINKHORN_BWD, _tile_kernels_sinkhorn_bwd = safe_import_from(
    "tile_kernels.mhc.sinkhorn_kernel",
    "_mhc_sinkhorn_bwd",
    msg="TileKernels low-level sinkhorn backward kernel is unavailable.",
)
_HAS_MILES_SPARSE_ATTN, _miles_sparse_attn_tilelang = safe_import_from(
    "nemo_automodel.components.models.deepseek_v4.kernels.sparse_attention",
    "sparse_attn_tilelang",
    msg="Vendored Miles DeepSeek V4 sparse attention is unavailable. Install tilelang to use backend.attn='tilelang'.",
)
_HAS_MILES_SPARSE_ATTN_CHUNKED, _miles_sparse_attn_tilelang_head_chunked = safe_import_from(
    "nemo_automodel.components.models.deepseek_v4.kernels.sparse_attention",
    "sparse_attn_tilelang_head_chunked",
    msg="Vendored Miles DeepSeek V4 chunked sparse attention is unavailable. Install tilelang to use "
    "backend.attn='tilelang'.",
)
_HAS_MILES_INDEXER, _miles_batched_indexer_fwd = safe_import_from(
    "nemo_automodel.components.models.deepseek_v4.kernels.tilelang_indexer_fwd",
    "batched_indexer_fwd",
    msg="Vendored Miles DeepSeek V4 indexer is unavailable. Install tilelang to use backend.attn='tilelang'.",
)
_HAS_MILES_CU_SEQLENS, _miles_make_causal_cu_seqlens = safe_import_from(
    "nemo_automodel.components.models.deepseek_v4.kernels.tilelang_indexer_fwd",
    "_make_causal_cu_seqlens",
    msg="Vendored Miles DeepSeek V4 indexer cu-seqlens helper is unavailable.",
)
_HAS_MILES_INDEXER_AUTOGRAD, _miles_v4_lighting_indexer = safe_import_from(
    "nemo_automodel.components.models.deepseek_v4.kernels.tilelang_indexer",
    "v4_lighting_indexer",
    msg="Vendored Miles DeepSeek V4 autograd indexer is unavailable.",
)


def _env_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def _env_optional_positive_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def _normalize_sparse_attn_head_chunk(total_heads: int, requested_heads: int) -> int:
    """Pick a TileLang sparse-attention head chunk that does not create ragged tiles.

    The vendored TileLang kernel pads the head tile to at least 16 lanes. Using
    8-head chunks with TP=1 creates underfilled head tiles and has been observed
    to hang on long DSV4 CP shapes. Prefer power-of-two chunks that divide the
    local head count.
    """
    if total_heads <= 16:
        return total_heads

    candidates = [16, 32, 64]
    valid = [heads for heads in candidates if heads <= total_heads and total_heads % heads == 0]
    if not valid:
        return total_heads
    if requested_heads in valid:
        return requested_heads

    smaller_or_equal = [heads for heads in valid if heads <= requested_heads]
    if smaller_or_equal:
        return max(smaller_or_equal)
    return min(valid)


def _pad_sparse_attn_kv_for_tilelang(kv: torch.Tensor, multiple: int = 64) -> torch.Tensor:
    """Pad KV length for TileLang sparse-attention shape stability.

    The sparse-attention kernel only reads keys listed in ``topk_idxs``. Under
    CP, nonzero ranks can expose sliding-window KV lengths such as 8319 or
    41087 that are not tile-aligned; padding KV to a 64 boundary avoids
    TileLang runtime hangs without changing attention semantics.
    """
    if multiple <= 1:
        return kv
    remainder = kv.shape[1] % multiple
    if remainder == 0:
        return kv
    pad_len = multiple - remainder
    pad = kv.new_zeros((kv.shape[0], pad_len, kv.shape[2]))
    return torch.cat((kv, pad), dim=1).contiguous()


def is_dsv4_kernel_available(name: Literal["sinkhorn", "sparse_attn", "indexer"]) -> bool:
    """Return whether the optional TileLang kernel package for ``name`` is importable."""
    if name == "sinkhorn":
        return _HAS_TILE_KERNELS_SINKHORN and _HAS_TILE_KERNELS_SINKHORN_FWD and _HAS_TILE_KERNELS_SINKHORN_BWD
    if name == "sparse_attn":
        return _HAS_MILES_SPARSE_ATTN
    if name == "indexer":
        return _HAS_MILES_INDEXER and _HAS_MILES_CU_SEQLENS and _HAS_MILES_INDEXER_AUTOGRAD
    raise ValueError(f"Unknown DeepSeek V4 kernel name: {name}")


def _all_cuda(*tensors: torch.Tensor) -> bool:
    return all(tensor.is_cuda for tensor in tensors)


def _should_use_tilelang(
    backend: str,
    *,
    available: bool,
    kernel_name: str,
    tensors: tuple[torch.Tensor, ...],
    require_bf16: bool = False,
) -> bool:
    if backend == "torch" or backend == "sparse_torch":
        return False

    can_run = available and _all_cuda(*tensors)
    if require_bf16:
        can_run = can_run and all(tensor.dtype == torch.bfloat16 for tensor in tensors)

    if backend == "tilelang" and not can_run:
        requirement = "CUDA bfloat16 tensors" if require_bf16 else "CUDA tensors"
        tensor_specs = ", ".join(
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} is_cuda={tensor.is_cuda}"
            for tensor in tensors
        )
        raise RuntimeError(
            f"dsv4 {kernel_name} TileLang backend was requested, but the optional kernel is unavailable "
            f"or inputs do not satisfy {requirement}. "
            f"available={available}; tensors=[{tensor_specs}]"
        )
    return backend == "tilelang" or (backend == "auto" and can_run)


def sinkhorn_normalize_torch(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    """Torch reference for TileKernels MHC Sinkhorn normalization."""
    x = x.softmax(dim=-1) + eps
    x = x / (x.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        x = x / (x.sum(dim=-2, keepdim=True) + eps)
    return x


class _Dsv4TileKernelsSinkhorn(torch.autograd.Function):
    """TileKernels Sinkhorn wrapper that accepts non-contiguous backward gradients."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        repeat: int,
        eps: float,
    ) -> torch.Tensor:
        flat_x = x.contiguous().view(-1, *x.shape[-2:])
        hidden_size = flat_x.shape[1]
        flat_output = torch.empty_like(flat_x)
        fwd_kernel = _tile_kernels_sinkhorn_fwd(hidden_size, 1, repeat, eps)
        bwd_kernel = _tile_kernels_sinkhorn_bwd(hidden_size, 32, repeat, eps)
        fwd_kernel(flat_x, flat_output)
        ctx.save_for_backward(flat_x)
        ctx.bwd_kernel = bwd_kernel
        ctx.input_shape = x.shape
        return flat_output.view_as(x)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        (flat_x,) = ctx.saved_tensors
        flat_grad_output = grad_output.contiguous().view_as(flat_x)
        pad_tokens = (-flat_x.shape[0]) % 32
        if pad_tokens:
            flat_x_for_bwd = torch.cat([flat_x, flat_x.new_zeros((pad_tokens, *flat_x.shape[1:]))], dim=0)
            grad_for_bwd = torch.cat(
                [flat_grad_output, flat_grad_output.new_zeros((pad_tokens, *flat_grad_output.shape[1:]))],
                dim=0,
            )
            grad_input_padded = torch.empty_like(flat_x_for_bwd)
            ctx.bwd_kernel(grad_for_bwd, flat_x_for_bwd, grad_input_padded)
            flat_grad_input = grad_input_padded[: flat_x.shape[0]]
        else:
            flat_grad_input = torch.empty_like(flat_x)
            ctx.bwd_kernel(flat_grad_output, flat_x, flat_grad_input)
        return flat_grad_input.view(ctx.input_shape), None, None


def _tile_kernels_sinkhorn_contiguous_grad(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    if _HAS_TILE_KERNELS_SINKHORN_FWD and _HAS_TILE_KERNELS_SINKHORN_BWD:
        return _Dsv4TileKernelsSinkhorn.apply(x, repeat, eps)
    return _tile_kernels_sinkhorn(x.contiguous(), repeat=repeat, eps=eps)


def dsv4_sinkhorn_normalize(
    x: torch.Tensor,
    *,
    backend: Dsv4SinkhornBackend,
    repeat: int,
    eps: float,
) -> torch.Tensor:
    """Normalize HyperConnection combination logits with torch or TileKernels."""
    if _should_use_tilelang(
        backend,
        available=is_dsv4_kernel_available("sinkhorn"),
        kernel_name="sinkhorn",
        tensors=(x,),
    ):
        return _tile_kernels_sinkhorn_contiguous_grad(x, repeat=repeat, eps=eps)
    return sinkhorn_normalize_torch(x, repeat=repeat, eps=eps)


def query_seq_positions_from_ids(query_seq_ids: torch.Tensor) -> torch.Tensor:
    """Return each query token's ordinal inside its packed sample."""
    query_seq_ids = query_seq_ids.to(dtype=torch.int64)
    valid = query_seq_ids >= 0
    if query_seq_ids.shape[1] == 0:
        return torch.empty_like(query_seq_ids, dtype=torch.int64)

    prev = torch.cat(
        [query_seq_ids.new_full((query_seq_ids.shape[0], 1), -1), query_seq_ids[:, :-1]],
        dim=1,
    )
    starts = valid & (query_seq_ids != prev)
    absolute = torch.arange(query_seq_ids.shape[1], device=query_seq_ids.device, dtype=torch.int64).view(1, -1)
    start_positions = torch.where(starts, absolute.expand_as(query_seq_ids), torch.zeros_like(query_seq_ids))
    start_offsets = start_positions.cummax(dim=1).values
    positions = absolute.expand_as(query_seq_ids) - start_offsets
    return torch.where(valid, positions, torch.full_like(positions, -1))


def _query_local_positions(
    query_seq_ids: torch.Tensor,
    query_positions: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if query_positions is None:
        return query_seq_positions_from_ids(query_seq_ids)
    query_positions = query_positions.to(device=device, dtype=torch.int64)
    if query_positions.ndim == 1:
        if query_positions.numel() != seq_len:
            raise ValueError(
                f"query_positions length must match seq_len (got {query_positions.numel()} vs {seq_len})"
            )
        return query_positions.unsqueeze(0).expand(batch_size, -1)
    if query_positions.shape != (batch_size, seq_len):
        raise ValueError(f"query_positions must have shape {(batch_size, seq_len)}, got {tuple(query_positions.shape)}")
    return query_positions


def _pooled_local_positions(
    pooled_seq_ids: torch.Tensor,
    pooled_positions: torch.Tensor | None,
    *,
    batch_size: int,
    pooled_len: int,
    device: torch.device,
) -> torch.Tensor:
    if pooled_positions is None:
        return pooled_seq_positions_from_ids(pooled_seq_ids)
    pooled_positions = pooled_positions.to(device=device, dtype=torch.int64)
    if pooled_positions.ndim == 1:
        if pooled_positions.numel() != pooled_len:
            raise ValueError(
                f"pooled_positions length must match pooled_len (got {pooled_positions.numel()} vs {pooled_len})"
            )
        return pooled_positions.unsqueeze(0).expand(batch_size, -1)
    if pooled_positions.shape != (batch_size, pooled_len):
        raise ValueError(
            f"pooled_positions must have shape {(batch_size, pooled_len)}, got {tuple(pooled_positions.shape)}"
        )
    return pooled_positions


def build_dsv4_sparse_topk_indices(
    *,
    batch_size: int,
    seq_len: int,
    key_len: int,
    window_size: int,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
    compress_ratio: int = 0,
    compressed_topk: torch.Tensor | None = None,
    n_pooled: int = 0,
    query_start: int = 0,
    query_global_start: int | None = None,
    query_positions: torch.Tensor | None = None,
    query_key_positions: torch.Tensor | None = None,
    query_local_positions: torch.Tensor | None = None,
    pooled_local_positions: torch.Tensor | None = None,
    raw_key_len: int | None = None,
    query_seq_ids: torch.Tensor | None = None,
    raw_key_seq_ids: torch.Tensor | None = None,
    pooled_seq_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build Miles-style top-k key indices for DSV4 local-window + compressed KV attention."""
    raw_key_len = key_len - n_pooled if raw_key_len is None else raw_key_len
    if n_pooled > 0 and compress_ratio <= 0:
        raise ValueError("compress_ratio must be positive when compressed KV entries are requested")
    query_global_start = query_start if query_global_start is None else query_global_start
    window = min(raw_key_len, window_size)
    if query_positions is None:
        q_global_pos = torch.arange(query_global_start, query_global_start + seq_len, device=device)
        q_pos = torch.arange(query_start, query_start + seq_len, device=device)
    else:
        q_global_pos = query_positions.to(device=device, dtype=torch.int64)
        if q_global_pos.numel() != seq_len:
            raise ValueError(
                f"query_positions length must match seq_len (got {q_global_pos.numel()} vs {seq_len})"
            )
        q_pos = (
            query_key_positions.to(device=device, dtype=torch.int64)
            if query_key_positions is not None
            else q_global_pos
        )
    k_pos = (q_pos.unsqueeze(1) - window_size + 1).clamp(min=0) + torch.arange(window, device=device)
    valid_window_2d = (k_pos <= q_pos.unsqueeze(1)) & (k_pos < raw_key_len)
    window_topk_2d = torch.where(valid_window_2d, k_pos, torch.full_like(k_pos, -1)).to(torch.int32)
    if query_seq_ids is not None and raw_key_seq_ids is not None:
        query_seq_ids = query_seq_ids.to(device=device, dtype=torch.int64)
        raw_key_seq_ids = raw_key_seq_ids.to(device=device, dtype=torch.int64)
        if query_seq_ids.shape != (batch_size, seq_len):
            raise ValueError(f"query_seq_ids must have shape {(batch_size, seq_len)}, got {tuple(query_seq_ids.shape)}")
        if raw_key_seq_ids.shape != (batch_size, raw_key_len):
            raise ValueError(
                f"raw_key_seq_ids must have shape {(batch_size, raw_key_len)}, got {tuple(raw_key_seq_ids.shape)}"
            )
        safe_raw = window_topk_2d.clamp(min=0, max=max(raw_key_len - 1, 0)).long()
        gathered_seq_ids = raw_key_seq_ids[:, safe_raw]
        same_seq = gathered_seq_ids == query_seq_ids.unsqueeze(-1)
        valid_window = (window_topk_2d.unsqueeze(0) >= 0) & same_seq & (query_seq_ids.unsqueeze(-1) >= 0)
        topk = torch.where(
            valid_window,
            window_topk_2d.unsqueeze(0).expand(batch_size, -1, -1),
            torch.full((batch_size, seq_len, window), -1, dtype=torch.int32, device=device),
        )
    else:
        topk = window_topk_2d.unsqueeze(0).expand(batch_size, -1, -1)

    if n_pooled > 0:
        if compressed_topk is not None:
            compressed_topk = compressed_topk.to(device=device)
            valid_pool_idx = (compressed_topk >= 0) & (compressed_topk < n_pooled)
            if query_seq_ids is not None and pooled_seq_ids is not None:
                pooled_seq_ids = pooled_seq_ids.to(device=device, dtype=torch.int64)
                if pooled_seq_ids.shape != (batch_size, n_pooled):
                    raise ValueError(
                        f"pooled_seq_ids must have shape {(batch_size, n_pooled)}, got {tuple(pooled_seq_ids.shape)}"
                    )
                safe_pooled = compressed_topk.clamp(min=0, max=max(n_pooled - 1, 0)).long()
                pooled_seq_for_query = torch.gather(
                    pooled_seq_ids.unsqueeze(1).expand(-1, seq_len, -1),
                    dim=-1,
                    index=safe_pooled,
                )
                query_local_pos = _query_local_positions(
                    query_seq_ids.to(device=device, dtype=torch.int64),
                    query_local_positions,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=device,
                )
                pooled_local_pos = _pooled_local_positions(
                    pooled_seq_ids,
                    pooled_local_positions,
                    batch_size=batch_size,
                    pooled_len=n_pooled,
                    device=device,
                )
                pooled_positions_for_query = torch.gather(
                    pooled_local_pos.unsqueeze(1).expand(-1, seq_len, -1),
                    dim=-1,
                    index=safe_pooled,
                )
                threshold = ((query_local_pos + 1) // compress_ratio).unsqueeze(-1)
                valid_compressed = (
                    valid_pool_idx
                    & (pooled_seq_for_query == query_seq_ids.to(device=device, dtype=torch.int64).unsqueeze(-1))
                    & (pooled_positions_for_query >= 0)
                    & (pooled_positions_for_query < threshold)
                    & (query_seq_ids.to(device=device, dtype=torch.int64).unsqueeze(-1) >= 0)
                )
                compressed_topk = torch.where(valid_pool_idx, compressed_topk, torch.full_like(compressed_topk, -1))
                compressed_topk = torch.where(valid_compressed, compressed_topk, torch.full_like(compressed_topk, -1))
            else:
                compressed_topk = torch.where(valid_pool_idx, compressed_topk, torch.full_like(compressed_topk, -1))
            compressed = torch.where(
                compressed_topk >= 0,
                compressed_topk + raw_key_len,
                torch.full_like(compressed_topk, -1),
            ).to(torch.int32)
        else:
            pooled_pos = torch.arange(n_pooled, device=device).unsqueeze(0).expand(seq_len, -1)
            if query_seq_ids is not None and pooled_seq_ids is not None:
                query_seq_ids = query_seq_ids.to(device=device, dtype=torch.int64)
                pooled_seq_ids = pooled_seq_ids.to(device=device, dtype=torch.int64)
                if pooled_seq_ids.shape != (batch_size, n_pooled):
                    raise ValueError(
                        f"pooled_seq_ids must have shape {(batch_size, n_pooled)}, got {tuple(pooled_seq_ids.shape)}"
                    )
                pooled_seq_positions = _pooled_local_positions(
                    pooled_seq_ids,
                    pooled_local_positions,
                    batch_size=batch_size,
                    pooled_len=n_pooled,
                    device=device,
                )
                query_local_pos = _query_local_positions(
                    query_seq_ids,
                    query_local_positions,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=device,
                )
                threshold = ((query_local_pos + 1) // compress_ratio).unsqueeze(-1)
                allowed = (
                    (pooled_seq_positions.unsqueeze(1) >= 0)
                    & (pooled_seq_positions.unsqueeze(1) < threshold)
                    & (pooled_seq_ids.unsqueeze(1) == query_seq_ids.unsqueeze(-1))
                    & (query_seq_ids.unsqueeze(-1) >= 0)
                )
                pooled_pos_expanded = pooled_pos.unsqueeze(0).expand(batch_size, -1, -1)
                compressed = torch.where(
                    allowed,
                    pooled_pos_expanded + raw_key_len,
                    torch.full_like(pooled_pos_expanded, -1),
                ).to(torch.int32)
            else:
                threshold = ((q_global_pos + 1) // compress_ratio).unsqueeze(1)
                allowed = pooled_pos < threshold
                compressed = torch.where(
                    allowed,
                    pooled_pos + raw_key_len,
                    torch.full_like(pooled_pos, -1),
                ).to(torch.int32).unsqueeze(0)
                compressed = compressed.expand(batch_size, -1, -1)
        topk = torch.cat([topk, compressed], dim=-1)

    if attention_mask is not None:
        if attention_mask.dim() != 4:
            raise ValueError(f"Expected 4D additive attention mask, got rank {attention_mask.dim()}")
        safe_topk = topk.clamp(min=0, max=key_len - 1)
        mask_values = torch.gather(attention_mask[:, 0, :, :key_len], dim=-1, index=safe_topk.long())
        topk = torch.where(mask_values < 0, torch.full_like(topk, -1), topk)

    valid_range = (topk >= 0) & (topk < key_len)
    return torch.where(valid_range, topk, torch.full_like(topk, -1))


def sparse_attention_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    sinks: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Miles sparse MQA torch reference.

    Args:
        q: Query tensor with shape ``[B, S, H, D]``.
        kv: Single-head KV tensor with shape ``[B, K, D]``.
        sinks: Per-head attention sink logits with shape ``[H]``.
        topk_idxs: Key indices with shape ``[B, S, K_top]``; ``-1`` masks an entry.
        sm_scale: Attention scaling factor.
    """
    q_float = q.float()
    kv_float = kv.float()
    batch, _, heads, _ = q.shape
    key_len = kv.shape[1]
    valid = (topk_idxs >= 0) & (topk_idxs < key_len)
    safe_idxs = topk_idxs.clamp(min=0, max=max(key_len - 1, 0))
    batch_idx = torch.arange(batch, device=q.device).view(batch, 1, 1)
    kv_gathered = kv_float[batch_idx, safe_idxs]

    scores = torch.einsum("bshd,bskd->bshk", q_float, kv_gathered) * sm_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    scores_max = scores.max(dim=-1).values.clamp(min=-1e30)
    exp_scores = torch.exp(scores - scores_max.unsqueeze(-1))

    numerator = torch.einsum("bshk,bskd->bshd", exp_scores, kv_gathered)
    denominator = exp_scores.sum(dim=-1) + torch.exp(sinks.float().view(1, 1, heads) - scores_max)
    return (numerator / denominator.unsqueeze(-1)).to(q.dtype)


def dense_attention_topk_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    sinks: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Dense torch oracle for the Miles top-k sparse-attention contract."""
    batch, seq_len, heads, _ = q.shape
    key_len = kv.shape[1]
    topk_len = topk_idxs.shape[-1]
    attn_mask = torch.zeros(batch, seq_len, key_len, dtype=torch.bool, device=q.device)
    valid = (topk_idxs >= 0) & (topk_idxs < key_len)
    batch_idx = torch.arange(batch, device=q.device).view(batch, 1, 1).expand(batch, seq_len, topk_len)
    seq_idx = torch.arange(seq_len, device=q.device).view(1, seq_len, 1).expand(batch, seq_len, topk_len)
    attn_mask[batch_idx[valid], seq_idx[valid], topk_idxs[valid].long()] = True

    scores = torch.einsum("bshd,bkd->bshk", q.float(), kv.float()) * sm_scale
    scores = scores.masked_fill(~attn_mask.unsqueeze(2), float("-inf"))
    scores_max = scores.max(dim=-1).values.clamp(min=-1e30)
    exp_scores = torch.exp(scores - scores_max.unsqueeze(-1))
    numerator = torch.einsum("bshk,bkd->bshd", exp_scores, kv.float())
    denominator = exp_scores.sum(dim=-1) + torch.exp(sinks.float().view(1, 1, heads) - scores_max)
    return (numerator / denominator.unsqueeze(-1)).to(q.dtype)


def _dsv4_sparse_attention_tilelang(
    q: torch.Tensor,
    kv: torch.Tensor,
    sinks: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float,
    max_heads_per_kernel: int,
) -> torch.Tensor:
    """Run one TileLang sparse-attention query chunk."""
    if topk_idxs.numel() == 0 or not bool((topk_idxs >= 0).any().item()):
        # Sink-only rows have zero numerator. Keep a zero-gradient dependency on
        # q so autograd sees the expected slice shape in chunked execution.
        return q * 0
    if q.shape[2] > max_heads_per_kernel:
        if not _HAS_MILES_SPARSE_ATTN_CHUNKED:
            raise RuntimeError("Chunked Miles DeepSeek V4 sparse attention is unavailable")
        # Do not materialize full transposed Q here. The chunked wrapper slices Q
        # and makes each small head tile contiguous before launching TileLang.
        return _miles_sparse_attn_tilelang_head_chunked(q, kv, sinks, topk_idxs, max_heads_per_kernel, sm_scale)
    return _miles_sparse_attn_tilelang(q.contiguous(), kv, sinks, topk_idxs, sm_scale)


def dsv4_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    sinks: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float,
    *,
    backend: Dsv4SparseAttentionBackend,
) -> torch.Tensor:
    """Run DSV4 sparse attention through Miles TileLang kernels or torch fallback."""
    valid_topk = (topk_idxs >= 0) & (topk_idxs < kv.shape[1])
    topk_idxs = torch.where(valid_topk, topk_idxs, torch.full_like(topk_idxs, -1))
    use_tilelang = _should_use_tilelang(
        backend,
        available=_HAS_MILES_SPARSE_ATTN,
        kernel_name="sparse attention",
        tensors=(q, kv),
        require_bf16=True,
    )
    if use_tilelang:
        kv = _pad_sparse_attn_kv_for_tilelang(kv.contiguous())
        sinks = sinks.float().contiguous()
        topk_idxs = topk_idxs.to(torch.int32).contiguous()

        # Miles runs this kernel under tensor parallelism, so the kernel sees a
        # small local head count. AutoModel's DSV4 recipe currently uses TP=1,
        # which would launch a single H=64, D=512 kernel with excessive shared
        # memory/register pressure. Chunking heads preserves the same TileLang
        # fwd/bwd kernels and lets autograd sum the per-chunk KV gradients.
        default_max_heads = 16 if q.shape[-1] >= 256 else 64
        requested_max_heads = _env_positive_int("DSV4_SPARSE_ATTN_MAX_HEADS_PER_KERNEL", default_max_heads)
        max_heads_per_kernel = _normalize_sparse_attn_head_chunk(q.shape[2], requested_max_heads)
        query_chunk_size = _env_optional_positive_int("DSV4_SPARSE_ATTN_QUERY_CHUNK")
        if query_chunk_size is not None and q.shape[1] > query_chunk_size:
            outputs = []
            for start in range(0, q.shape[1], query_chunk_size):
                end = min(start + query_chunk_size, q.shape[1])
                outputs.append(
                    _dsv4_sparse_attention_tilelang(
                        q[:, start:end, :, :],
                        kv,
                        sinks,
                        topk_idxs[:, start:end, :],
                        sm_scale,
                        max_heads_per_kernel,
                    )
                )
            return torch.cat(outputs, dim=1)
        return _dsv4_sparse_attention_tilelang(q, kv, sinks, topk_idxs, sm_scale, max_heads_per_kernel)
    return sparse_attention_torch(q, kv, sinks, topk_idxs.long(), sm_scale)


def indexer_scores_torch(
    q: torch.Tensor,
    pooled_kv: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Torch reference for the Miles DSV4 C4 indexer score kernel."""
    scores = torch.matmul(q.float(), pooled_kv.transpose(-1, -2).float().unsqueeze(1))
    scores = torch.relu(scores) * softmax_scale
    return (scores * weights.float().unsqueeze(-1)).sum(dim=2)


def _make_global_causal_cu_seqlens(
    seq_len_q: int,
    seq_len_kv: int,
    compress_ratio: int,
    device: torch.device,
    query_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build compressed-KV causal ranges for local CP query shards.

    Miles' original helper assumes local query positions start at zero.  Under
    context parallelism each rank scores a local query shard against global
    pooled KV, so the end offset must be derived from global positions.
    """
    positions = torch.arange(
        int(query_start),
        int(query_start) + int(seq_len_q),
        device=device,
        dtype=torch.int64,
    )
    cu_seqlen_ks = torch.zeros(seq_len_q, device=device, dtype=torch.int32)
    cu_seqlen_ke = ((positions + 1) // int(compress_ratio)).clamp(min=0, max=seq_len_kv).to(torch.int32)
    return cu_seqlen_ks, cu_seqlen_ke


def pooled_seq_positions_from_ids(pooled_seq_ids: torch.Tensor) -> torch.Tensor:
    """Return each pooled token's ordinal inside its packed sample.

    ``pooled_seq_ids`` uses ``-1`` for invalid/padding windows and otherwise
    contains contiguous packed sample ids.  DeepSeek V4 causal compressed-KV
    checks must compare query positions against this per-sample ordinal.  The
    ordinal is based on the original pooled slot offset inside the sample, so
    invalid overlap windows do not renumber later valid pooled slots.
    """
    pooled_seq_ids = pooled_seq_ids.to(dtype=torch.int64)
    valid = pooled_seq_ids >= 0
    if pooled_seq_ids.shape[1] == 0:
        return torch.empty_like(pooled_seq_ids, dtype=torch.int64)

    absolute = torch.arange(pooled_seq_ids.shape[1], device=pooled_seq_ids.device, dtype=torch.int64)
    positions = torch.full_like(pooled_seq_ids, -1)
    # Fallback path only; packed DSV4 mainline passes explicit pooled positions.
    # Preserve invalid holes instead of renumbering later slots in the same sample.
    for batch_idx in range(pooled_seq_ids.shape[0]):
        row = pooled_seq_ids[batch_idx]
        for seq_id in torch.unique(row[valid[batch_idx]]):
            seq_mask = row == seq_id
            first = absolute[seq_mask][0]
            positions[batch_idx, seq_mask] = absolute[seq_mask] - first
    return positions


def _mask_indexer_scores_causal_(
    scores: torch.Tensor,
    *,
    compress_ratio: int,
    query_start: int = 0,
    query_positions: torch.Tensor | None = None,
    query_local_positions: torch.Tensor | None = None,
    query_seq_ids: torch.Tensor | None = None,
    pooled_seq_ids: torch.Tensor | None = None,
    pooled_local_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the same compressed-KV causal rule to dense torch indexer scores."""
    seq_len_q = scores.shape[1]
    seq_len_kv = scores.shape[-1]
    if query_positions is None:
        query_positions = torch.arange(
            int(query_start),
            int(query_start) + int(seq_len_q),
            device=scores.device,
            dtype=torch.int64,
        )
    else:
        query_positions = query_positions.to(device=scores.device, dtype=torch.int64)
        if query_positions.numel() != seq_len_q:
            raise ValueError(
                f"query_positions length must match query score length (got {query_positions.numel()} vs {seq_len_q})"
            )
    if query_seq_ids is not None and pooled_seq_ids is not None:
        query_seq_ids = query_seq_ids.to(device=scores.device, dtype=torch.int64)
        pooled_seq_ids = pooled_seq_ids.to(device=scores.device, dtype=torch.int64)
        if query_seq_ids.shape != scores.shape[:2]:
            raise ValueError(f"query_seq_ids must have shape {tuple(scores.shape[:2])}, got {tuple(query_seq_ids.shape)}")
        if pooled_seq_ids.shape != (scores.shape[0], seq_len_kv):
            raise ValueError(
                f"pooled_seq_ids must have shape {(scores.shape[0], seq_len_kv)}, got {tuple(pooled_seq_ids.shape)}"
            )
        query_local_pos = _query_local_positions(
            query_seq_ids,
            query_local_positions,
            batch_size=scores.shape[0],
            seq_len=seq_len_q,
            device=scores.device,
        )
        threshold = ((query_local_pos + 1) // int(compress_ratio)).unsqueeze(-1)
        pooled_positions = _pooled_local_positions(
            pooled_seq_ids,
            pooled_local_positions,
            batch_size=scores.shape[0],
            pooled_len=seq_len_kv,
            device=scores.device,
        ).unsqueeze(1)
        scores = scores.masked_fill_(pooled_positions >= threshold, float("-inf"))
        same_seq = pooled_seq_ids.unsqueeze(1) == query_seq_ids.unsqueeze(-1)
        scores = scores.masked_fill(
            (query_seq_ids.unsqueeze(-1) < 0) | (pooled_positions < 0) | (~same_seq),
            float("-inf"),
        )
    else:
        threshold = ((query_positions + 1) // int(compress_ratio)).view(1, -1, 1)
        pooled_positions = torch.arange(seq_len_kv, device=scores.device, dtype=torch.int64).view(1, 1, -1)
        scores = scores.masked_fill_(pooled_positions >= threshold, float("-inf"))
    return scores


def _query_positions_are_contiguous(query_positions: torch.Tensor | None, query_start: int, seq_len: int) -> bool:
    if query_positions is None:
        return True
    expected = torch.arange(
        int(query_start),
        int(query_start) + int(seq_len),
        device=query_positions.device,
        dtype=query_positions.dtype,
    )
    return bool(torch.equal(query_positions.reshape(-1), expected))


def extract_indexer_topk_scores_torch(logits: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    """Extract top-k score values, masking invalid entries with ``-inf``."""
    key_len = logits.shape[-1]
    valid = (topk_indices >= 0) & (topk_indices < key_len)
    safe_indices = topk_indices.clamp(min=0, max=max(key_len - 1, 0)).to(torch.int64)
    scores = torch.gather(logits, dim=-1, index=safe_indices)
    return torch.where(valid, scores, torch.full((), float("-inf"), dtype=scores.dtype, device=scores.device))


def dsv4_indexer_scores(
    q: torch.Tensor,
    pooled_kv: torch.Tensor,
    weights: torch.Tensor,
    *,
    compress_ratio: int,
    softmax_scale: float,
    backend: Dsv4IndexerBackend,
    query_start: int = 0,
    query_positions: torch.Tensor | None = None,
    query_local_positions: torch.Tensor | None = None,
    query_seq_ids: torch.Tensor | None = None,
    pooled_seq_ids: torch.Tensor | None = None,
    pooled_local_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run DSV4 C4 indexer scores through Miles TileLang kernels or torch fallback."""
    supports_tilelang_cu_seqlens = (
        _query_positions_are_contiguous(query_positions, query_start, q.shape[1])
        and query_seq_ids is None
        and pooled_seq_ids is None
    )
    effective_backend = backend if supports_tilelang_cu_seqlens else "torch"
    if _should_use_tilelang(
        effective_backend,
        available=_HAS_MILES_INDEXER and _HAS_MILES_CU_SEQLENS and supports_tilelang_cu_seqlens,
        kernel_name="indexer",
        tensors=(q, pooled_kv),
        require_bf16=True,
    ):
        seq_len = q.shape[1]
        seq_len_kv = pooled_kv.shape[1]
        cu_ks, cu_ke = _make_global_causal_cu_seqlens(
            seq_len,
            seq_len_kv,
            compress_ratio,
            q.device,
            query_start=query_start,
        )
        return _miles_batched_indexer_fwd(
            q.transpose(0, 1).contiguous(),
            pooled_kv.transpose(0, 1).contiguous(),
            (weights * softmax_scale).transpose(0, 1).contiguous(),
            cu_ks,
            cu_ke,
        )
    return _mask_indexer_scores_causal_(
        indexer_scores_torch(q, pooled_kv, weights, softmax_scale),
        compress_ratio=compress_ratio,
        query_start=query_start,
        query_positions=query_positions,
        query_local_positions=query_local_positions,
        query_seq_ids=query_seq_ids,
        pooled_seq_ids=pooled_seq_ids,
        pooled_local_positions=pooled_local_positions,
    )


def streaming_indexer_topk_indices_torch(
    q: torch.Tensor,
    pooled_kv: torch.Tensor,
    weights: torch.Tensor,
    *,
    topk: int,
    compress_ratio: int,
    softmax_scale: float,
    query_start: int = 0,
    query_positions: torch.Tensor | None = None,
    query_local_positions: torch.Tensor | None = None,
    query_seq_ids: torch.Tensor | None = None,
    pooled_seq_ids: torch.Tensor | None = None,
    pooled_local_positions: torch.Tensor | None = None,
    query_block_size: int = 256,
    key_block_size: int = 2048,
) -> torch.Tensor:
    """Memory-bounded top-k for the DSV4 C4 indexer.

    Dense indexer logits have shape ``[B, S_query, S_pooled]`` and become the
    first 128K blocker.  This helper streams over query and pooled-KV blocks,
    keeping only the running top-k per query while preserving the same scoring
    expression as :func:`indexer_scores_torch`.
    """
    batch, seq_len, _, _ = q.shape
    pooled_len = pooled_kv.shape[1]
    topk = min(int(topk), pooled_len)
    if topk <= 0:
        return torch.empty(batch, seq_len, 0, dtype=torch.int32, device=q.device)

    query_block_size = max(1, int(query_block_size))
    key_block_size = max(1, int(key_block_size))
    output = torch.empty(batch, seq_len, topk, dtype=torch.int32, device=q.device)
    if query_positions is not None:
        query_positions = query_positions.to(device=q.device, dtype=torch.int64)
        if query_positions.numel() != seq_len:
            raise ValueError(
                f"query_positions length must match query length (got {query_positions.numel()} vs {seq_len})"
            )
    if query_seq_ids is not None:
        query_seq_ids = query_seq_ids.to(device=q.device, dtype=torch.int64)
        if query_seq_ids.shape != (batch, seq_len):
            raise ValueError(f"query_seq_ids must have shape {(batch, seq_len)}, got {tuple(query_seq_ids.shape)}")
    if query_local_positions is not None and query_seq_ids is not None:
        query_local_positions = _query_local_positions(
            query_seq_ids,
            query_local_positions,
            batch_size=batch,
            seq_len=seq_len,
            device=q.device,
        )
    if pooled_seq_ids is not None:
        pooled_seq_ids = pooled_seq_ids.to(device=q.device, dtype=torch.int64)
        if pooled_seq_ids.shape != (batch, pooled_len):
            raise ValueError(f"pooled_seq_ids must have shape {(batch, pooled_len)}, got {tuple(pooled_seq_ids.shape)}")
        pooled_seq_positions = _pooled_local_positions(
            pooled_seq_ids,
            pooled_local_positions,
            batch_size=batch,
            pooled_len=pooled_len,
            device=q.device,
        )
    else:
        pooled_seq_positions = None

    for query_begin in range(0, seq_len, query_block_size):
        query_end = min(query_begin + query_block_size, seq_len)
        q_chunk = q[:, query_begin:query_end]
        weights_chunk = weights[:, query_begin:query_end].float()
        query_positions_chunk = (
            torch.arange(
                query_start + query_begin,
                query_start + query_end,
                device=q.device,
            )
            if query_positions is None
            else query_positions[query_begin:query_end]
        )
        query_seq_ids_chunk = None if query_seq_ids is None else query_seq_ids[:, query_begin:query_end]
        if query_seq_ids_chunk is not None and pooled_seq_ids is not None:
            if query_local_positions is None:
                query_local_positions_chunk = query_seq_positions_from_ids(query_seq_ids)[:, query_begin:query_end]
            else:
                query_local_positions_chunk = query_local_positions[:, query_begin:query_end]
            threshold = ((query_local_positions_chunk + 1) // compress_ratio).unsqueeze(-1)
        else:
            threshold = ((query_positions_chunk + 1) // compress_ratio).view(1, -1, 1)

        running_scores = None
        running_indices = None
        for key_begin in range(0, pooled_len, key_block_size):
            key_end = min(key_begin + key_block_size, pooled_len)
            kv_chunk = pooled_kv[:, key_begin:key_end]
            scores = torch.matmul(q_chunk.float(), kv_chunk.transpose(-1, -2).float().unsqueeze(1))
            scores = torch.relu(scores) * softmax_scale
            scores = (scores * weights_chunk.unsqueeze(-1)).sum(dim=2)

            if query_seq_ids_chunk is not None and pooled_seq_ids is not None:
                pooled_seq_ids_chunk = pooled_seq_ids[:, key_begin:key_end].unsqueeze(1)
                pooled_positions = pooled_seq_positions[:, key_begin:key_end].unsqueeze(1)
                scores = scores.masked_fill(pooled_positions >= threshold, float("-inf"))
                same_seq = pooled_seq_ids_chunk == query_seq_ids_chunk.unsqueeze(-1)
                scores = scores.masked_fill(
                    (query_seq_ids_chunk.unsqueeze(-1) < 0) | (pooled_positions < 0) | (~same_seq),
                    float("-inf"),
                )
            else:
                pooled_positions = torch.arange(key_begin, key_end, device=q.device).view(1, 1, -1)
                scores = scores.masked_fill(pooled_positions >= threshold, float("-inf"))
            block_topk = min(topk, key_end - key_begin)
            block_scores, block_indices = scores.topk(block_topk, dim=-1)
            block_indices = block_indices + key_begin

            if running_scores is None:
                running_scores = block_scores
                running_indices = block_indices
            else:
                merged_scores = torch.cat([running_scores, block_scores], dim=-1)
                merged_indices = torch.cat([running_indices, block_indices], dim=-1)
                keep = min(topk, merged_scores.shape[-1])
                running_scores, order = merged_scores.topk(keep, dim=-1)
                running_indices = torch.gather(merged_indices, dim=-1, index=order)

        if running_scores.shape[-1] < topk:
            pad_len = topk - running_scores.shape[-1]
            running_scores = torch.cat(
                [running_scores, running_scores.new_full((*running_scores.shape[:-1], pad_len), float("-inf"))],
                dim=-1,
            )
            running_indices = torch.cat(
                [running_indices, running_indices.new_full((*running_indices.shape[:-1], pad_len), -1)],
                dim=-1,
            )
        running_indices = torch.where(
            running_scores == float("-inf"),
            torch.full_like(running_indices, -1),
            running_indices,
        )
        output[:, query_begin:query_end] = running_indices.to(torch.int32)

    return output


def dsv4_indexer_topk_scores(
    q: torch.Tensor,
    pooled_kv: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    compress_ratio: int,
    softmax_scale: float,
    backend: Dsv4IndexerBackend,
) -> torch.Tensor:
    """Run DSV4 C4 top-k indexer scores through Miles autograd kernels or torch fallback."""
    pooled_len = pooled_kv.shape[1]
    valid_topk = (topk_indices >= 0) & (topk_indices < pooled_len)
    safe_topk_indices = torch.where(valid_topk, topk_indices, torch.full_like(topk_indices, -1))
    if _should_use_tilelang(
        backend,
        available=_HAS_MILES_INDEXER_AUTOGRAD,
        kernel_name="indexer autograd",
        tensors=(q, pooled_kv),
        require_bf16=True,
    ):
        scores, _ = _miles_v4_lighting_indexer(
            q.transpose(0, 1).contiguous(),
            pooled_kv.transpose(0, 1).contiguous(),
            (weights * softmax_scale).transpose(0, 1).contiguous(),
            compress_ratio,
            safe_topk_indices.shape[-1],
            safe_topk_indices.to(torch.int32).contiguous(),
        )
        return scores
    logits = indexer_scores_torch(q, pooled_kv, weights, softmax_scale)
    return extract_indexer_topk_scores_torch(logits, safe_topk_indices.long())
