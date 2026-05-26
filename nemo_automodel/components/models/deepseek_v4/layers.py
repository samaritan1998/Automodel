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

"""DeepSeek V4 Attention Layer.

Architecture (from official inference/model.py):

Q path:
  x  -> wq_a [hidden -> q_lora_rank]
     -> q_norm (RMSNorm)
     -> wq_b  [q_lora_rank -> n_heads * head_dim]
     -> reshape [n_heads, head_dim]
     -> per-head RMSNorm  (q_norm applied per-head in official code)
     -> apply_rotary_emb on last rope_head_dim dims

KV path (K = V, single latent):
  x  -> wkv   [hidden -> head_dim]        # single KV head, K = V = kv
     -> kv_norm (RMSNorm on head_dim)
     -> apply_rotary_emb on last rope_head_dim dims
  K = V = kv  (one latent vector serves both key and value)

Output path (grouped):
  o [bsz, seq, n_heads, head_dim]
    -> reshape [bsz, seq, n_groups, n_heads_per_group * head_dim]
    -> wo_a einsum per group: [n_heads_per_group * head_dim] -> [o_lora_rank]
    -> reshape [bsz, seq, n_groups * o_lora_rank]
    -> wo_b [n_groups * o_lora_rank -> hidden]

attn_sink: learnable per-head scalar bias added to attention-sink position score.

HC (Hyper-Connections):
  Each Block maintains hc_mult=4 copies of the hidden state.
  hc_pre  reduces [bsz, seq, hc_mult, dim] -> [bsz, seq, dim] via Sinkhorn mixing.
  hc_post expands [bsz, seq, dim] -> [bsz, seq, hc_mult, dim].
  See ``DeepseekV4HyperConnection.compute_weights`` and
  ``optimized_kernels.dsv4_sinkhorn_normalize`` for the torch reference and
  optional TileKernels Sinkhorn path.

Sliding-window attention, compressed KV pooling, sparse compressed-index
selection, and attention sinks are implemented for training.  The KV-cache
inference path is intentionally left out.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.optimized_kernels import (
    _query_positions_are_contiguous,
    build_dsv4_sparse_topk_indices,
    dsv4_indexer_scores,
    dsv4_sinkhorn_normalize,
    dsv4_sparse_attention,
    pooled_seq_positions_from_ids,
    query_seq_positions_from_ids,
    streaming_indexer_topk_indices_torch,
)


def _dsv4_kernel_backend(backend: BackendConfig) -> str:
    """Use TileLang DSV4 kernels only when the attention backend requests them."""
    return "tilelang" if backend.attn == "tilelang" else "torch"


def _dsv4_sinkhorn_backend(backend: BackendConfig) -> str:
    """Sinkhorn is optional even when sparse attention must stay on TileLang."""
    return "auto" if backend.attn == "tilelang" else "torch"


def _dsv4_hc_collapse(pre: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Collapse HC streams without materializing ``pre[..., None] * x``."""
    dtype = x.dtype
    hidden = x.shape[-1]
    hc_mult = x.shape[-2]
    collapsed = torch.bmm(
        pre.to(dtype).reshape(-1, 1, hc_mult),
        x.reshape(-1, hc_mult, hidden),
    ).squeeze(1)
    return collapsed.reshape(*x.shape[:-2], hidden)


def _dsv4_env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _dsv4_debug_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return torch.distributed.get_rank()
        except Exception:
            pass
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def _dsv4_debug_selected(selector: str, value: int | None) -> bool:
    selector = selector.strip().lower()
    if selector in {"", "*", "all"} or value is None:
        return True
    for part in selector.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start, end = part.split("-", 1)
                if int(start) <= value <= int(end):
                    return True
            elif int(part) == value:
                return True
        except ValueError:
            continue
    return False


def _dsv4_debug_value(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)},dtype={value.dtype},device={value.device},requires_grad={value.requires_grad})"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_dsv4_debug_value(v) for v in value) + "]"
    return repr(value)


def _dsv4_debug(stage: str, layer_idx: int | None = None, **fields: Any) -> None:
    if not _dsv4_env_flag("DSV4_CP_DEBUG"):
        return
    rank = _dsv4_debug_rank()
    if not _dsv4_debug_selected(os.environ.get("DSV4_CP_DEBUG_RANKS", "all"), rank):
        return
    if not _dsv4_debug_selected(os.environ.get("DSV4_CP_DEBUG_LAYERS", "all"), layer_idx):
        return
    local_rank = os.environ.get("LOCAL_RANK", "?")
    payload = " ".join(f"{key}={_dsv4_debug_value(value)}" for key, value in fields.items())
    print(
        f"[DSV4_CP_DEBUG rank={rank} local_rank={local_rank} layer={layer_idx if layer_idx is not None else '-'}] "
        f"{stage} {payload}",
        file=sys.stderr,
        flush=True,
    )


def _dsv4_debug_backward_hook(
    tensor: torch.Tensor | None,
    stage: str,
    layer_idx: int | None = None,
    **fields: Any,
) -> None:
    if not _dsv4_env_flag("DSV4_CP_DEBUG_BACKWARD"):
        return
    if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
        return

    _dsv4_debug(f"{stage}.backward_hook.register", layer_idx, tensor=tensor, **fields)

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        _dsv4_debug(f"{stage}.backward", layer_idx, grad=grad, **fields)
        if _dsv4_env_flag("DSV4_SYNC_IN_BACKWARD_HOOK") and grad.is_cuda:
            torch.cuda.synchronize(grad.device)
            _dsv4_debug(f"{stage}.backward.sync", layer_idx, grad=grad, **fields)
        return grad

    tensor.register_hook(_hook)


def _dsv4_zero_dependency(*values: Any) -> torch.Tensor | None:
    zero = None

    def _visit(value: Any) -> None:
        nonlocal zero
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                term = value.sum() * 0.0
                zero = term if zero is None else zero + term
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                _visit(item)

    for value in values:
        _visit(value)
    return zero


def _dsv4_attach_zero_dependency(tensor: torch.Tensor, *values: Any) -> torch.Tensor:
    zero = _dsv4_zero_dependency(*values)
    if zero is None:
        return tensor
    return tensor + zero.to(device=tensor.device, dtype=tensor.dtype)


def _dsv4_validate_topk_indices(topk_idxs: torch.Tensor, key_len: int, layer_idx: int | None) -> None:
    if not _dsv4_env_flag("DSV4_CP_DEBUG_VALIDATE_TOPK"):
        return
    if topk_idxs.numel() == 0:
        return
    topk_min = int(topk_idxs.amin().item())
    topk_max = int(topk_idxs.amax().item())
    _dsv4_debug("attn.sparse_topk.validate", layer_idx, topk_min=topk_min, topk_max=topk_max, key_len=key_len)
    if topk_min < -1 or topk_max >= key_len:
        raise RuntimeError(
            f"DeepSeek V4 sparse topk out of range at layer {layer_idx}: "
            f"min={topk_min}, max={topk_max}, key_len={key_len}"
        )


class _ChunkedInplaceRMSNormNoWeight(torch.autograd.Function):
    """Memory-bounded no-weight RMSNorm for large Q tensors.

    The naive ``x * rsqrt(mean(x.square()))`` materializes a full-size square
    temporary.  At 128K/CP4, DSV4 Q is roughly 2 GiB per rank, so that
    temporary alone can OOM.  This function normalizes in-place over sequence
    chunks and saves only the normalized output plus per-token inverse RMS.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float, chunk_size: int) -> torch.Tensor:
        ctx.mark_dirty(x)
        chunk_size = max(1, int(chunk_size))
        inv_rms = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)
        for begin in range(0, x.shape[1], chunk_size):
            end = min(begin + chunk_size, x.shape[1])
            x_chunk = x[:, begin:end]
            inv_chunk = torch.rsqrt(x_chunk.float().square().mean(dim=-1, keepdim=True) + eps)
            x_chunk.mul_(inv_chunk.to(dtype=x_chunk.dtype))
            inv_rms[:, begin:end].copy_(inv_chunk)
        ctx.chunk_size = chunk_size
        ctx.save_for_backward(x, inv_rms)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        y, inv_rms = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        grad_input = torch.empty_like(grad_output)
        for begin in range(0, y.shape[1], chunk_size):
            end = min(begin + chunk_size, y.shape[1])
            grad_chunk = grad_output[:, begin:end]
            y_chunk = y[:, begin:end]
            mean_grad_y = (grad_chunk * y_chunk).mean(dim=-1, keepdim=True)
            grad_input[:, begin:end].copy_(
                (grad_chunk - y_chunk * mean_grad_y) * inv_rms[:, begin:end].to(dtype=grad_chunk.dtype)
            )
        return grad_input, None, None


def _chunked_q_rms_norm_(q: torch.Tensor, eps: float) -> torch.Tensor:
    try:
        chunk_size = int(os.environ.get("DSV4_Q_RMS_CHUNK_SIZE", "512"))
    except ValueError:
        chunk_size = 512
    if chunk_size <= 0:
        return q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)
    return _ChunkedInplaceRMSNormNoWeight.apply(q, float(eps), chunk_size)


def _cp_mesh_enabled(cp_mesh) -> bool:
    return (
        cp_mesh is not None
        and cp_mesh.size() > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )


def _cp_all_gather(
    tensor: torch.Tensor,
    cp_mesh,
    dim: int,
    *,
    debug_layer_idx: int | None = None,
    debug_label: str | None = None,
) -> torch.Tensor:
    if not _cp_mesh_enabled(cp_mesh):
        return tensor
    from torch.distributed.nn.functional import all_gather

    if debug_label is not None:
        _dsv4_debug(
            f"{debug_label}.all_gather.before",
            debug_layer_idx,
            tensor=tensor,
            dim=dim,
            cp_size=cp_mesh.size(),
            cp_rank=_cp_mesh_rank(cp_mesh),
        )
        _dsv4_debug_backward_hook(tensor, f"{debug_label}.all_gather.input", debug_layer_idx)
    parts = all_gather(tensor.contiguous(), group=cp_mesh.get_group())
    result = torch.cat(tuple(parts), dim=dim).contiguous()
    if debug_label is not None:
        _dsv4_debug(f"{debug_label}.all_gather.after", debug_layer_idx, result=result)
        _dsv4_debug_backward_hook(result, f"{debug_label}.all_gather.result", debug_layer_idx)
    return result


def _cp_mesh_rank(cp_mesh) -> int:
    if not _cp_mesh_enabled(cp_mesh):
        return 0
    return torch.distributed.get_rank(group=cp_mesh.get_group())


def _dsv4_cp_layout(value: Any = None) -> str:
    layout = value if value is not None else os.environ.get("DSV4_CP_LAYOUT", "contiguous")
    return str(layout).strip().lower().replace("-", "_")


def _cp_layout_is_zigzag(layout: str) -> bool:
    return layout in {"zigzag", "dual_chunk", "dual_chunk_swap", "dualchunkswap"}


def _split_zigzag_halves(tensor: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    local_len = tensor.shape[dim]
    if local_len % 2 != 0:
        raise ValueError(f"DeepSeek V4 zigzag CP requires even local sequence length, got {local_len}")
    half_len = local_len // 2
    return tensor.narrow(dim, 0, half_len), tensor.narrow(dim, half_len, half_len)


def _cat_zigzag_global_order(first_parts: tuple[torch.Tensor, ...], second_parts: tuple[torch.Tensor, ...], dim: int):
    return torch.cat((*first_parts, *reversed(second_parts)), dim=dim).contiguous()


def _cp_all_gather_zigzag_halves(
    tensor: torch.Tensor,
    cp_mesh,
    dim: int,
    *,
    debug_layer_idx: int | None = None,
    debug_label: str | None = None,
) -> torch.Tensor:
    if not _cp_mesh_enabled(cp_mesh):
        return tensor
    from torch.distributed.nn.functional import all_gather

    first, second = _split_zigzag_halves(tensor, dim)
    if debug_label is not None:
        _dsv4_debug(
            f"{debug_label}.zigzag_all_gather.before",
            debug_layer_idx,
            first=first,
            second=second,
            dim=dim,
            cp_size=cp_mesh.size(),
            cp_rank=_cp_mesh_rank(cp_mesh),
        )
        _dsv4_debug_backward_hook(first, f"{debug_label}.zigzag_all_gather.first_input", debug_layer_idx)
        _dsv4_debug_backward_hook(second, f"{debug_label}.zigzag_all_gather.second_input", debug_layer_idx)
    first_parts = tuple(all_gather(first.contiguous(), group=cp_mesh.get_group()))
    second_parts = tuple(all_gather(second.contiguous(), group=cp_mesh.get_group()))
    result = _cat_zigzag_global_order(first_parts, second_parts, dim)
    if debug_label is not None:
        _dsv4_debug(f"{debug_label}.zigzag_all_gather.after", debug_layer_idx, result=result)
        _dsv4_debug_backward_hook(result, f"{debug_label}.zigzag_all_gather.result", debug_layer_idx)
    return result


def _query_positions_1d(position_ids: torch.Tensor | None, seq_len: int, device: torch.device, fallback_start: int):
    if position_ids is None:
        return torch.arange(fallback_start, fallback_start + seq_len, device=device, dtype=torch.int64)
    if position_ids.ndim != 2 or position_ids.shape[1] != seq_len:
        raise ValueError(f"DeepSeek V4 CP expects 2D position_ids with seq_len={seq_len}, got {tuple(position_ids.shape)}")
    return position_ids[0].to(device=device, dtype=torch.int64).contiguous()


def _cp_gather_sliding_window_kv(
    kv: torch.Tensor,
    cp_mesh,
    window_size: int,
    *,
    cp_layout: str = "contiguous",
    debug_layer_idx: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Gather only the raw KV span needed by sliding-window attention."""
    if not _cp_mesh_enabled(cp_mesh):
        return kv, 0
    if _cp_layout_is_zigzag(cp_layout):
        full_kv = _cp_all_gather_zigzag_halves(
            kv,
            cp_mesh,
            dim=2,
            debug_layer_idx=debug_layer_idx,
            debug_label="attn.raw_kv.zigzag_full",
        )
        return full_kv, 0
    cp_rank = _cp_mesh_rank(cp_mesh)
    local_seq_len = kv.shape[2]
    tail_len = max(0, min(window_size - 1, local_seq_len))
    _dsv4_debug(
        "attn.raw_kv.tail.begin",
        debug_layer_idx,
        kv=kv,
        window_size=window_size,
        tail_len=tail_len,
        cp_rank=cp_rank,
        cp_size=cp_mesh.size(),
    )
    if tail_len == 0:
        return kv, cp_rank * local_seq_len
    if tail_len < window_size - 1:
        return (
            _cp_all_gather(kv, cp_mesh, dim=2, debug_layer_idx=debug_layer_idx, debug_label="attn.raw_kv.short"),
            0,
        )

    tail = kv[:, :, -tail_len:, :].contiguous()
    from torch.distributed.nn.functional import all_gather

    _dsv4_debug("attn.raw_kv.tail_all_gather.before", debug_layer_idx, tail=tail)
    tails = tuple(all_gather(tail, group=cp_mesh.get_group()))
    _dsv4_debug("attn.raw_kv.tail_all_gather.after", debug_layer_idx, gathered_parts=len(tails))
    if cp_rank == 0:
        prefix = kv[:, :, :0, :]
        raw_start = 0
    else:
        prefix = tails[cp_rank - 1]
        raw_start = cp_rank * local_seq_len - tail_len
    gathered = torch.cat([prefix, kv], dim=2).contiguous()
    return _dsv4_attach_zero_dependency(gathered, tails), raw_start


def _cp_gather_sliding_window_metadata(
    tensor: torch.Tensor,
    cp_mesh,
    window_size: int,
    *,
    cp_layout: str = "contiguous",
) -> tuple[torch.Tensor, int]:
    """Gather token metadata with the same raw-KV span used for sparse attention."""
    if not _cp_mesh_enabled(cp_mesh):
        return tensor, 0
    if _cp_layout_is_zigzag(cp_layout):
        return _cp_all_gather_zigzag_halves(tensor, cp_mesh, dim=1), 0

    cp_rank = _cp_mesh_rank(cp_mesh)
    local_seq_len = tensor.shape[1]
    tail_len = max(0, min(window_size - 1, local_seq_len))
    if tail_len == 0:
        return tensor, cp_rank * local_seq_len
    if tail_len < window_size - 1:
        return _cp_all_gather(tensor, cp_mesh, dim=1), 0

    from torch.distributed.nn.functional import all_gather

    tail = tensor[:, -tail_len:].contiguous()
    tails = tuple(all_gather(tail, group=cp_mesh.get_group()))
    if cp_rank == 0:
        prefix = tensor[:, :0]
        raw_start = 0
    else:
        prefix = tails[cp_rank - 1]
        raw_start = cp_rank * local_seq_len - tail_len
    return torch.cat([prefix, tensor], dim=1).contiguous(), raw_start


def _gather_full_cp_metadata(tensor: torch.Tensor | None, cp_mesh, *, cp_layout: str) -> torch.Tensor | None:
    if tensor is None:
        return None
    if not _cp_mesh_enabled(cp_mesh):
        return tensor
    if _cp_layout_is_zigzag(cp_layout):
        return _cp_all_gather_zigzag_halves(tensor, cp_mesh, dim=1)
    return _cp_all_gather(tensor, cp_mesh, dim=1)


def _pad_metadata_to_len(
    tensor: torch.Tensor | None,
    target_len: int,
    *,
    value: int,
) -> torch.Tensor | None:
    if tensor is None or tensor.shape[1] >= target_len:
        return tensor
    return F.pad(tensor, (0, target_len - tensor.shape[1]), value=value)


def _pool_seq_ids(
    seq_ids: torch.Tensor | None,
    ratio: int,
    pooled_len: int | None = None,
    *,
    overlap: bool = False,
) -> torch.Tensor | None:
    if seq_ids is None or ratio <= 0:
        return None
    usable = (seq_ids.shape[1] // ratio) * ratio
    if usable == 0:
        pooled = seq_ids.new_empty((seq_ids.shape[0], 0))
    else:
        windows = seq_ids[:, :usable].view(seq_ids.shape[0], usable // ratio, ratio)
        first = windows[:, :, 0]
        valid_current = (first >= 0) & (windows == first.unsqueeze(-1)).all(dim=-1)
        valid = valid_current
        pooled = torch.where(valid, first, torch.full_like(first, -1)).contiguous()
    if pooled_len is not None:
        pooled = pooled[:, :pooled_len].contiguous()
    return pooled


def _pool_position_ids(
    position_ids: torch.Tensor | None,
    ratio: int,
    pooled_len: int | None = None,
) -> torch.Tensor | None:
    """Return pooled-token causal ordinals derived from per-token position_ids."""
    if position_ids is None or ratio <= 0:
        return None
    usable = (position_ids.shape[1] // ratio) * ratio
    if usable == 0:
        pooled = position_ids.new_empty((position_ids.shape[0], 0))
    else:
        pooled = (position_ids[:, :usable:ratio] // ratio).contiguous()
    if pooled_len is not None:
        pooled = pooled[:, :pooled_len].contiguous()
    return pooled


def _validate_packed_pool_alignment(seq_ids: torch.Tensor | None, ratio: int) -> None:
    if seq_ids is None or ratio <= 1:
        return
    for batch_idx in range(seq_ids.shape[0]):
        row = seq_ids[batch_idx].to(torch.long)
        valid = row >= 0
        if not bool(valid.any()):
            continue
        prev = torch.cat([row.new_full((1,), -1), row[:-1]])
        starts = valid & (row != prev)
        start_positions = torch.nonzero(starts, as_tuple=False).flatten()
        if start_positions.numel() and bool((start_positions % ratio != 0).any()):
            bad_start = int(start_positions[start_positions % ratio != 0][0].item())
            raise ValueError(
                "DeepSeek V4 packed compressed-KV pooling requires each packed sample to start on a "
                f"compress_ratio boundary (ratio={ratio}, first bad start={bad_start}). "
                "Use THD packing with seq_padding_multiple >= the largest DSV4 compress ratio."
            )


def _build_indexer_topk_compressed_mask(
    attention_mask: torch.Tensor,
    indexer_topk: torch.Tensor,
    n_pooled: int,
) -> torch.Tensor:
    """Build ``[B, S, P]`` additive mask from indexer-selected compressed slots."""
    batch, _, seq_len, _ = attention_mask.shape
    min_val = torch.finfo(attention_mask.dtype).min
    if n_pooled <= 0:
        return torch.empty(batch, seq_len, 0, dtype=attention_mask.dtype, device=attention_mask.device)
    valid = (indexer_topk >= 0) & (indexer_topk < n_pooled)
    safe_idx = indexer_topk.clamp(min=0, max=n_pooled - 1).to(torch.int64)
    indicator = torch.zeros(
        (batch, seq_len, n_pooled),
        dtype=torch.int32,
        device=attention_mask.device,
    )
    indicator.scatter_add_(-1, safe_idx, valid.to(indicator.dtype))
    return torch.where(
        indicator > 0,
        torch.zeros((), dtype=attention_mask.dtype, device=attention_mask.device),
        torch.full((), min_val, dtype=attention_mask.dtype, device=attention_mask.device),
    )


def _pad_raw_kv_before_compressed_for_tilelang(
    raw_kv: torch.Tensor,
    *,
    multiple: int = 64,
    debug_layer_idx: int | None = None,
) -> torch.Tensor:
    """Pad the raw-KV segment so compressed KV starts on a stable tile boundary.

    TileLang sparse attention indexes raw-window rows and compressed rows from
    one concatenated KV tensor. Under CP, nonzero ranks have raw lengths such
    as 8319 because of the previous-rank sliding-window tail. Padding the raw
    segment before concatenating compressed KV keeps the compressed base aligned
    while leaving all top-k-referenced rows unchanged.
    """
    if multiple <= 1:
        return raw_kv
    remainder = raw_kv.shape[2] % multiple
    if remainder == 0:
        return raw_kv
    pad_len = multiple - remainder
    pad = raw_kv.new_zeros((*raw_kv.shape[:2], pad_len, raw_kv.shape[-1]))
    padded = torch.cat((raw_kv, pad), dim=2).contiguous()
    _dsv4_debug(
        "attn.raw_kv.pad_for_tilelang.after",
        debug_layer_idx,
        raw_kv=raw_kv,
        padded=padded,
        pad_len=pad_len,
        multiple=multiple,
    )
    return padded


def _cp_previous_rank_tail(
    tensor: torch.Tensor,
    cp_mesh,
    tail_len: int,
    *,
    debug_layer_idx: int | None = None,
    debug_label: str | None = None,
) -> torch.Tensor:
    if not _cp_mesh_enabled(cp_mesh) or tail_len <= 0:
        return tensor[:, :0, :]
    from torch.distributed.nn.functional import all_gather

    tail = tensor[:, -tail_len:, :].contiguous()
    if debug_label is not None:
        _dsv4_debug(f"{debug_label}.previous_tail.all_gather.before", debug_layer_idx, tail=tail)
    tails = tuple(all_gather(tail, group=cp_mesh.get_group()))
    if debug_label is not None:
        _dsv4_debug(f"{debug_label}.previous_tail.all_gather.after", debug_layer_idx, gathered_parts=len(tails))
    cp_rank = _cp_mesh_rank(cp_mesh)
    if cp_rank == 0:
        prefix = tensor[:, :0, :]
    else:
        prefix = tails[cp_rank - 1]
    return _dsv4_attach_zero_dependency(prefix, tails)


def _pool_projected_windows(
    kv: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    kv_norm: nn.Module,
    rotary: nn.Module,
    *,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    overlap: bool,
    start_pos: int,
    pool_position_ids: torch.Tensor | None = None,
    pool_seq_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if kv.shape != gate.shape:
        raise ValueError(f"DeepSeek V4 projected kv/gate shape mismatch: {kv.shape} vs {gate.shape}")
    usable = (kv.shape[1] // ratio) * ratio
    ready_kv = kv[:, :usable]
    ready_gate = gate[:, :usable]
    pooled = _pool_windows(
        ready_kv,
        ready_gate,
        ape,
        ratio,
        head_dim,
        overlap=overlap,
        seq_ids=None if pool_seq_ids is None else pool_seq_ids[:, :usable],
    )
    norm_weight = getattr(kv_norm, "weight", None)
    if norm_weight is not None:
        pooled = pooled.to(norm_weight.dtype)
    pooled = kv_norm(pooled)
    if pooled.shape[1] > 0:
        if pool_position_ids is None:
            positions = _rope_pool_positions(pooled.shape[1], start_pos, ratio, pooled.device, pooled.shape[0])
        else:
            pool_position_ids = pool_position_ids.to(device=pooled.device, dtype=torch.int64)
            positions = pool_position_ids[:, :usable:ratio][:, : pooled.shape[1]].contiguous()
        cos, sin = rotary(pooled, positions)
        pooled = _apply_partial_rope(pooled.unsqueeze(1), cos, sin, rope_head_dim).squeeze(1)
    return pooled


def _cp_pool_projected_and_gather(
    hidden_states: torch.Tensor,
    pooler,
    rotary: nn.Module,
    cp_mesh,
    local_start: int,
    *,
    cp_layout: str = "contiguous",
    debug_layer_idx: int | None = None,
    debug_label: str = "attn.cp_pool",
):
    if not _cp_mesh_enabled(cp_mesh):
        return None
    ratio = int(pooler.compress_ratio)
    _dsv4_debug(
        f"{debug_label}.begin",
        debug_layer_idx,
        hidden_states=hidden_states,
        ratio=ratio,
        local_start=local_start,
        overlap=getattr(pooler, "overlap", None),
    )
    if ratio <= 0 or hidden_states.shape[1] % ratio != 0:
        _dsv4_debug(f"{debug_label}.skip_unaligned", debug_layer_idx, seq_len=hidden_states.shape[1], ratio=ratio)
        return None

    hidden_states_fp32 = hidden_states.float()
    kv = pooler.wkv(hidden_states_fp32)
    gate = pooler.wgate(hidden_states_fp32)
    _dsv4_debug(f"{debug_label}.project.after", debug_layer_idx, kv=kv, gate=gate)

    if _cp_layout_is_zigzag(cp_layout):
        if hidden_states.shape[1] % 2 != 0:
            raise ValueError("DeepSeek V4 zigzag CP pooling requires even local sequence length")
        half_len = hidden_states.shape[1] // 2
        if half_len % ratio != 0:
            _dsv4_debug(
                f"{debug_label}.zigzag.skip_unaligned_half",
                debug_layer_idx,
                half_len=half_len,
                ratio=ratio,
            )
            return None

        cp_rank = _cp_mesh_rank(cp_mesh)
        cp_size = cp_mesh.size()
        low_start = cp_rank * half_len
        high_start = (2 * cp_size - cp_rank - 1) * half_len
        kv_low, kv_high = _split_zigzag_halves(kv, dim=1)
        gate_low, gate_high = _split_zigzag_halves(gate, dim=1)

        def _empty_prefix(tensor: torch.Tensor) -> torch.Tensor:
            return tensor[:, :0, :]

        prefix_len = min(ratio, half_len) if getattr(pooler, "overlap", False) else 0
        if prefix_len > 0:
            from torch.distributed.nn.functional import all_gather

            kv_low_tail = kv_low[:, -prefix_len:, :].contiguous()
            kv_high_tail = kv_high[:, -prefix_len:, :].contiguous()
            gate_low_tail = gate_low[:, -prefix_len:, :].contiguous()
            gate_high_tail = gate_high[:, -prefix_len:, :].contiguous()
            kv_low_tails = tuple(all_gather(kv_low_tail, group=cp_mesh.get_group()))
            kv_high_tails = tuple(all_gather(kv_high_tail, group=cp_mesh.get_group()))
            gate_low_tails = tuple(all_gather(gate_low_tail, group=cp_mesh.get_group()))
            gate_high_tails = tuple(all_gather(gate_high_tail, group=cp_mesh.get_group()))
            tail_keepalive = _dsv4_zero_dependency(
                kv_low_tails,
                kv_high_tails,
                gate_low_tails,
                gate_high_tails,
            )

            kv_low_prefix = _empty_prefix(kv_low) if cp_rank == 0 else kv_low_tails[cp_rank - 1]
            gate_low_prefix = _empty_prefix(gate_low) if cp_rank == 0 else gate_low_tails[cp_rank - 1]
            if cp_rank == cp_size - 1:
                kv_high_prefix = kv_low_tails[cp_rank]
                gate_high_prefix = gate_low_tails[cp_rank]
            else:
                kv_high_prefix = kv_high_tails[cp_rank + 1]
                gate_high_prefix = gate_high_tails[cp_rank + 1]
        else:
            kv_low_prefix = _empty_prefix(kv_low)
            kv_high_prefix = _empty_prefix(kv_high)
            gate_low_prefix = _empty_prefix(gate_low)
            gate_high_prefix = _empty_prefix(gate_high)
            tail_keepalive = None

        def _pool_segment(
            segment_kv: torch.Tensor,
            segment_gate: torch.Tensor,
            prefix_kv: torch.Tensor,
            prefix_gate: torch.Tensor,
            segment_start: int,
            label: str,
        ) -> torch.Tensor:
            drop_pools = 0
            pool_start = segment_start
            if prefix_kv.shape[1] > 0:
                segment_kv = torch.cat([prefix_kv, segment_kv], dim=1)
                segment_gate = torch.cat([prefix_gate, segment_gate], dim=1)
                pool_start = segment_start - prefix_kv.shape[1]
                drop_pools = prefix_kv.shape[1] // ratio
            _dsv4_debug(
                f"{debug_label}.zigzag.{label}.local_pool.before",
                debug_layer_idx,
                kv=segment_kv,
                gate=segment_gate,
                pool_start=pool_start,
                drop_pools=drop_pools,
            )
            pooled = pooler.pool_projected(segment_kv, segment_gate, rotary, start_pos=pool_start)
            if drop_pools:
                pooled = pooled[:, drop_pools:]
            _dsv4_debug(f"{debug_label}.zigzag.{label}.local_pool.after", debug_layer_idx, pooled=pooled)
            return pooled

        low_pooled = _pool_segment(kv_low, gate_low, kv_low_prefix, gate_low_prefix, low_start, "low")
        high_pooled = _pool_segment(kv_high, gate_high, kv_high_prefix, gate_high_prefix, high_start, "high")
        if tail_keepalive is not None:
            low_pooled = low_pooled + tail_keepalive.to(device=low_pooled.device, dtype=low_pooled.dtype)
            high_pooled = high_pooled + tail_keepalive.to(device=high_pooled.device, dtype=high_pooled.dtype)
        _dsv4_debug_backward_hook(low_pooled, f"{debug_label}.zigzag.low_pooled", debug_layer_idx)
        _dsv4_debug_backward_hook(high_pooled, f"{debug_label}.zigzag.high_pooled", debug_layer_idx)

        from torch.distributed.nn.functional import all_gather

        low_parts = tuple(all_gather(low_pooled.contiguous(), group=cp_mesh.get_group()))
        high_parts = tuple(all_gather(high_pooled.contiguous(), group=cp_mesh.get_group()))
        result = _cat_zigzag_global_order(low_parts, high_parts, dim=1)
        _dsv4_debug(f"{debug_label}.zigzag.pooled.after", debug_layer_idx, result=result)
        _dsv4_debug_backward_hook(result, f"{debug_label}.zigzag.pooled", debug_layer_idx)
        return result

    pool_start = local_start
    drop_pools = 0

    if pooler.overlap:
        prefix_kv = _cp_previous_rank_tail(
            kv, cp_mesh, ratio, debug_layer_idx=debug_layer_idx, debug_label=f"{debug_label}.kv"
        )
        prefix_gate = _cp_previous_rank_tail(
            gate, cp_mesh, ratio, debug_layer_idx=debug_layer_idx, debug_label=f"{debug_label}.gate"
        )
        if prefix_kv.shape[1] > 0:
            kv = torch.cat([prefix_kv, kv], dim=1)
            gate = torch.cat([prefix_gate, gate], dim=1)
            pool_start = local_start - ratio
            drop_pools = 1
        else:
            kv = _dsv4_attach_zero_dependency(kv, prefix_kv, prefix_gate)
        _dsv4_debug(
            f"{debug_label}.overlap.after",
            debug_layer_idx,
            kv=kv,
            gate=gate,
            pool_start=pool_start,
            drop_pools=drop_pools,
        )

    _dsv4_debug(f"{debug_label}.local_pool.before", debug_layer_idx, kv=kv, gate=gate, pool_start=pool_start)
    local_pooled = pooler.pool_projected(kv, gate, rotary, start_pos=pool_start)
    if drop_pools:
        local_pooled = local_pooled[:, drop_pools:]
    _dsv4_debug(f"{debug_label}.local_pool.after", debug_layer_idx, local_pooled=local_pooled)
    _dsv4_debug_backward_hook(local_pooled, f"{debug_label}.local_pooled", debug_layer_idx)
    return _cp_all_gather(
        local_pooled,
        cp_mesh,
        dim=1,
        debug_layer_idx=debug_layer_idx,
        debug_label=f"{debug_label}.pooled",
    )


def _first_position(position_ids: torch.Tensor | None, fallback: int) -> int:
    if position_ids is None:
        return fallback
    return int(position_ids.reshape(-1)[0].item())


# ---------------------------------------------------------------------------
# DeepSeek V4 attention + compressor + indexer + rotary embedding, ported
# verbatim from HuggingFace transformers PR 45616 (Arthur Zucker, "Add
# DeepSeek V4") at
#   transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py
# with two adjustments:
#   1) Rotary helper ``apply_rotary_pos_emb`` and ``repeat_kv`` are inlined
#      so we do not depend on HF's transformers-version-specific rotary API.
#   2) The ``DeepseekV4Cache`` integration is replaced with a minimal
#      training-only shim — KAutomodel training never carries a KV cache,
#      so ``accumulate_windows`` / ``update_pool`` are pass-throughs on a
#      per-forward scratch dict.  The KV-cache path is left for a future
#      inference port.
# The compressor + indexer modules are only constructed when a layer's
# ``compress_ratio`` is non-zero; sparse attention and indexer execution are
# selected through ``BackendConfig``.
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input (Llama / GPT-NeoX style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_partial_rope_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_head_dim: int
) -> torch.Tensor:
    """Interleaved RoPE on the last ``rope_head_dim`` dims of ``x`` (pairs are
    ``(2k, 2k+1)``).  Matches the DeepSeek inference reference's complex-mul
    formulation in ``dsv4flash/inference/model.py:apply_rotary_emb``: the
    released DSV4-Flash weights were trained with this layout, NOT the
    Llama-style ``rotate_half`` layout HF transformers PR 45616/45643 still
    uses (pairs ``(d, d+rd/2)``).

    Args:
        x: ``[..., rope_head_dim]`` (or larger trailing dim with rope on the
            last ``rope_head_dim`` slice).  Typical attention-layout shapes:
            ``[B, H, S, D]`` for q/k or ``[B, 1, S, D]`` for shared-KV.
        cos, sin: shape ``[B, S, rope_head_dim]`` produced by the Llama-style
            ``cat([freqs, freqs], -1)`` rotary; we take the first half which
            contains the unique per-pair frequencies (the second half is a
            duplicate that the Llama-style helper needs and we don't).
        rope_head_dim: Must be even.

    Inverse rotation: pass ``-sin`` instead of ``sin`` (caller's
    responsibility — same as our existing inverse-rope call site).
    """
    rd = rope_head_dim
    half = rd // 2
    input_dtype = x.dtype
    nope, rope = x[..., :-rd], x[..., -rd:]
    # Pair-reshape last dim: [..., rd] -> [..., rd/2, 2]
    rope_pairs = rope.float().unflatten(-1, (-1, 2))
    a, b = rope_pairs[..., 0], rope_pairs[..., 1]  # [..., rd/2]
    c = cos[..., :half].float()
    s = sin[..., :half].float()
    # Broadcast c/s up to ``a``'s rank by inserting a head dim before S.
    while c.ndim < a.ndim:
        c = c.unsqueeze(1)
        s = s.unsqueeze(1)
    new_a = a * c - b * s
    new_b = a * s + b * c
    new_rope = torch.stack([new_a, new_b], dim=-1).flatten(-2).to(input_dtype)
    return torch.cat([nope, new_rope], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Port of transformers.models.llama.modeling_llama.apply_rotary_pos_emb."""
    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Port of transformers.models.llama.modeling_llama.repeat_kv."""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _yarn_correction_dim(num_rotations: float, dim: int, base: float, max_seq_len: int) -> float:
    import math

    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _yarn_correction_range(low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int) -> tuple[int, int]:
    import math

    low = math.floor(_yarn_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_yarn_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp(min_v: float, max_v: float, dim: int, device=None) -> torch.Tensor:
    if min_v == max_v:
        max_v += 0.001
    linear = (torch.arange(dim, dtype=torch.float32, device=device) - min_v) / (max_v - min_v)
    return torch.clamp(linear, 0, 1)


class DeepseekV4RotaryEmbedding(nn.Module):
    """V4 rotary embedding.  Produces ``(cos, sin)`` sized to ``qk_rope_head_dim``
    (via ``partial_rotary_factor = qk_rope_head_dim / head_dim``), matching HF.

    YaRN is applied only when ``rope_scaling`` is provided.  DSV4 Flash uses it
    for the compress-rope path, while the main sliding-window rope keeps
    ``rope_scaling=None``.
    """

    inv_freq: torch.Tensor

    def __init__(
        self,
        rope_theta: float,
        head_dim: int,
        partial_rotary_factor: float,
        attention_scaling: float = 1.0,
        device: torch.device | None = None,
        rope_scaling: dict | None = None,
    ):
        super().__init__()
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        rope_type = str((rope_scaling or {}).get("type", (rope_scaling or {}).get("rope_type", ""))).lower()
        if rope_scaling and rope_type == "yarn":
            factor = float(rope_scaling.get("factor", 1.0))
            orig = int(rope_scaling.get("original_max_position_embeddings", 0))
            beta_fast = float(rope_scaling.get("beta_fast", 32))
            beta_slow = float(rope_scaling.get("beta_slow", 1))
            if orig > 0 and factor > 0:
                low, high = _yarn_correction_range(beta_fast, beta_slow, dim, rope_theta, orig)
                smooth = 1.0 - _yarn_linear_ramp(low, high, dim // 2, device=inv_freq.device)
                inv_freq = inv_freq / factor * (1.0 - smooth) + inv_freq * smooth
        self.attention_scaling = attention_scaling
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force fp32 for numerical stability.
        with torch.autocast(device_type=x.device.type if x.device.type != "mps" else "cpu", enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear (HF PR 45616 port).

    ``weight`` parameter has the standard ``nn.Linear`` shape
    ``[out_features, in_features_per_group]`` so quantizers keyed on
    ``nn.Linear.weight`` still find it; ``forward`` does per-group bmm.
    """

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., n_groups, in_features_per_group]
        batch_shape = x.shape[:-2]
        d_in = x.shape[-1]
        out_per_group = self.out_features // self.n_groups
        w = self.weight.view(self.n_groups, out_per_group, d_in)
        x = x.reshape(-1, self.n_groups, d_in).permute(1, 0, 2)
        y = torch.bmm(x, w.transpose(-1, -2)).permute(1, 0, 2)
        return y.reshape(*batch_shape, self.n_groups, out_per_group)


class DeepseekV4TrainCache:
    """Training-only cache shim mirroring the three methods ``DeepseekV4Compressor``
    / ``DeepseekV4Indexer`` call on ``DeepseekV4Cache``.

    KAutomodel training forward is stateless — we never persist KV or compressor
    windows across calls.  Each ``DeepseekV4Attention.forward`` creates a fresh
    cache instance, which holds per-layer scratch dicts for the duration of the
    call.  When a full window hasn't accumulated yet we return an empty tensor
    and let the downstream code handle it.
    """

    def __init__(self):
        self.compressor_state: list[dict] = []
        self.indexer_state: list[dict] = []

    def _branch_state(self, state_key: str, layer_idx: int) -> dict:
        store = getattr(self, state_key, None)
        if store is None:
            store = []
            setattr(self, state_key, store)
        while len(store) <= layer_idx:
            store.append({"buffer_kv": None, "buffer_gate": None, "pooled": None})
        return store[layer_idx]

    def accumulate_windows(
        self,
        kv: torch.Tensor,
        gate: torch.Tensor,
        layer_idx: int,
        state_key: str,
        ratio: int,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        state = self._branch_state(state_key, layer_idx)
        buf_kv, buf_gate = state["buffer_kv"], state["buffer_gate"]
        if buf_kv is not None and buf_kv.shape[1]:
            kv = torch.cat([buf_kv, kv], dim=1)
            gate = torch.cat([buf_gate, gate], dim=1)
        usable = (kv.shape[1] // ratio) * ratio
        state["buffer_kv"] = kv[:, usable:]
        state["buffer_gate"] = gate[:, usable:]
        pool_base = max(0, start_pos) - (buf_kv.shape[1] if buf_kv is not None else 0)
        return kv[:, :usable], gate[:, :usable], pool_base

    def update_pool(self, new_pooled: torch.Tensor, layer_idx: int, state_key: str) -> torch.Tensor:
        state = self._branch_state(state_key, layer_idx)
        pool = state["pooled"]
        if new_pooled.shape[1] > 0:
            pool = new_pooled if pool is None else torch.cat([pool, new_pooled], dim=1)
            state["pooled"] = pool
        if pool is None:
            pool = new_pooled.new_zeros((new_pooled.shape[0], 0, new_pooled.shape[-1]))
        return pool


def _apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_head_dim: int) -> torch.Tensor:
    """Split ``x`` along its last dim into nope (first) and rope (last
    ``rope_head_dim``) slices, rotate only the rope slice with INTERLEAVED
    pair-RoPE (pairs ``(2k, 2k+1)``), concat back.

    The DSV4-Flash released checkpoint uses interleaved RoPE end-to-end
    (see ``dsv4flash/inference/model.py:apply_rotary_emb`` — complex
    multiplication on ``view_as_complex`` of pairs).  HF transformers PR
    45616 / PR 45643 ship a Llama-style ``rotate_half`` here instead, which
    pairs ``(d, d+rd/2)``.  Same algebra but a different dim-to-frequency
    mapping — the released weights expect the interleaved layout, so the
    Llama-style helper produces wrong activations on the released checkpoint
    (verified empirically: kv_post_rope cosine drops from 0.9999 to 0.866
    after one block under Llama-style; matches at >0.999 under interleaved).
    """
    return _apply_partial_rope_interleaved(x, cos, sin, rope_head_dim)


def _overlap_transform(tensor: torch.Tensor, head_dim: int, fill_value: float) -> torch.Tensor:
    """Reshape ``[B, S, ratio, 2*head_dim]`` -> ``[B, S, 2*ratio, head_dim]`` with the
    cross-window overlap from the DeepSeek inference reference (``Compressor.overlap_transform``
    in ``dsv4flash/inference/model.py:307-314``).

    Window N consumes:
      * positions ``[ratio:]`` of the new tensor: the **second half** of the feature dim
        of window N (current block).
      * positions ``[:ratio]`` of the new tensor: the **first half** of the feature dim
        of window N-1 (previous block, i.e. the overlap into the past).

    Window 0 has no previous block, so its ``[:ratio]`` slice is left at ``fill_value``
    (``0`` for the kv tensor, ``-inf`` for the score tensor so softmax masks it out).
    """
    b, s, ratio, _ = tensor.shape
    new = tensor.new_full((b, s, 2 * ratio, head_dim), fill_value)
    new[:, :, ratio:] = tensor[:, :, :, head_dim:]
    new[:, 1:, :ratio] = tensor[:, :-1, :, :head_dim]
    return new


def _pool_windows(
    kv: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    ratio: int,
    head_dim: int,
    overlap: bool = False,
    seq_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax-gated sum-pool over ``ratio`` consecutive tokens.

    Non-overlap mode (HF PR 45616 layout, ratio==128 in V4-Flash):
      Input  ``kv``/``gate`` of shape ``[B, length, head_dim]``.
      Reshape to ``[B, length/ratio, ratio, head_dim]`` and pool over the ``ratio`` axis.

    Overlap mode (DeepSeek inference reference layout, ratio==4 in V4-Flash):
      Input  ``kv``/``gate`` of shape ``[B, length, 2*head_dim]`` (``wkv``/``wgate``
      project to ``2*head_dim`` so each window can carry both its own kv and a
      half-overlap into the next window).
      Reshape to ``[B, length/ratio, ratio, 2*head_dim]``, apply :func:`_overlap_transform`
      to remap to ``[B, length/ratio, 2*ratio, head_dim]``, then pool over the ``2*ratio``
      axis.  Each compressed token thus aggregates ``2*ratio = 8`` raw tokens — the
      ``ratio`` tokens of the current window plus the ``ratio`` tokens of the previous
      window — giving smoother compression boundaries that the released checkpoint
      was trained under.

    HF PR 45616 omits the overlap path entirely; the released DSV4-Flash safetensors
    have ``ape``/``wkv``/``wgate`` shapes that only match the overlap layout (``[ratio,
    2*head_dim]`` and ``[2*head_dim, hidden]``), so we must support it here to load
    the released weights.
    """
    coff = 2 if overlap else 1
    feat = coff * head_dim
    batch, length, _ = kv.shape
    n_windows = length // ratio
    kv_w = kv.view(batch, n_windows, ratio, feat)
    gate_w = gate.view(batch, n_windows, ratio, feat) + ape
    if overlap:
        kv_w = _overlap_transform(kv_w, head_dim, fill_value=0.0)
        gate_w = _overlap_transform(gate_w, head_dim, fill_value=float("-inf"))
        if seq_ids is not None and n_windows > 0:
            seq_windows = seq_ids[:, : n_windows * ratio].view(batch, n_windows, ratio).to(device=kv.device)
            first = seq_windows[:, :, 0]
            valid_current = (first >= 0) & (seq_windows == first.unsqueeze(-1)).all(dim=-1)
            prev_first = torch.cat([first.new_full((batch, 1), -1), first[:, :-1]], dim=1)
            prev_valid = torch.cat(
                [torch.zeros((batch, 1), dtype=torch.bool, device=kv.device), valid_current[:, :-1]],
                dim=1,
            )
            use_previous = (valid_current & prev_valid & (prev_first == first)).view(batch, n_windows, 1, 1)
            kv_w[:, :, :ratio] = torch.where(use_previous, kv_w[:, :, :ratio], torch.zeros_like(kv_w[:, :, :ratio]))
            gate_w[:, :, :ratio] = torch.where(
                use_previous,
                gate_w[:, :, :ratio],
                torch.full_like(gate_w[:, :, :ratio], float("-inf")),
            )
    return (kv_w * gate_w.softmax(dim=2)).sum(dim=2)


def _rope_pool_positions(
    pool_length: int, pool_base: int, ratio: int, device: torch.device, batch: int
) -> torch.Tensor:
    return (torch.arange(pool_length, device=device) * ratio + pool_base).unsqueeze(0).expand(batch, -1)


def build_causal_padding_mask(
    attention_mask: torch.Tensor | None,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int = 1,
    sliding_window: int | None = None,
) -> torch.Tensor | None:
    """Build a 4D additive causal+padding (+optional sliding-window) mask
    compatible with ``eager_attention_with_sink``.

    Mirrors HF's ``create_sliding_window_causal_mask`` (used in
    ``DeepseekV4Model.forward``): each query at position ``i`` attends only to
    keys at positions ``[max(0, i - sliding_window + 1), i]``.  The DSV4-Flash
    weights were trained with this banding on every layer, so dropping it makes
    the softmax see a different distribution than training and degrades loss.

    Inputs:
        attention_mask: 2D ``[B, S]`` tensor with 1=valid, 0=padding (HF convention),
            or already-4D additive mask, or ``None``.
        sliding_window: if not None, mask out keys further back than this many
            positions from the query (in addition to causal masking).
    Returns:
        ``[B, 1, S, S]`` additive mask of ``dtype`` (0 where keep, large negative
        where mask).
    """
    min_value = torch.finfo(dtype).min if dtype.is_floating_point else -1e9
    causal = torch.full((seq_len, seq_len), min_value, dtype=dtype, device=device)
    causal = torch.triu(causal, diagonal=1)
    if sliding_window is not None and sliding_window > 0:
        # Mask k_pos < q_pos - (sliding_window - 1)  →  diagonal = -(window - 1)
        # tril at that diagonal keeps the lower-band; we need to MASK the lower
        # tail (older keys).  Build a "too old" mask: positions where (q - k) >= window.
        idx = torch.arange(seq_len, device=device)
        too_old = (idx.unsqueeze(0) - idx.unsqueeze(1)) >= sliding_window  # [k_pos, q_pos] ?
        # We want shape [q_pos, k_pos], so use [q_pos=row, k_pos=col]:
        too_old = (idx.unsqueeze(1) - idx.unsqueeze(0)) >= sliding_window
        causal = causal.masked_fill(too_old, min_value)
    causal = causal.unsqueeze(0).unsqueeze(0)  # [1,1,S,S]
    if attention_mask is None:
        return causal.expand(batch_size, 1, seq_len, seq_len).contiguous()
    if attention_mask.dim() == 4:
        return attention_mask.to(dtype)
    if attention_mask.dim() == 2:
        # 1=valid, 0=padding -> 0 keep, min_value mask, broadcast over query rows
        pad_add = (1.0 - attention_mask.to(dtype)) * min_value  # [B, S]
        pad_add = pad_add.unsqueeze(1).unsqueeze(2)  # [B,1,1,S]
        return (causal + pad_add).to(dtype)
    raise ValueError(f"Unsupported attention_mask rank: {attention_mask.dim()}")


def build_packed_causal_padding_mask(
    seq_lens: torch.Tensor,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Build a 4D additive block-causal mask from packed-sequence lengths."""
    if seq_lens.dim() == 1:
        seq_lens = seq_lens.unsqueeze(0)
    seq_lens = seq_lens.to(device=device, dtype=torch.long)
    seq_lens = torch.where(seq_lens > 0, seq_lens, torch.zeros((), device=device, dtype=torch.long))

    batch_size = seq_lens.shape[0]
    positions = torch.arange(seq_len, device=device, dtype=torch.long).expand(batch_size, -1)
    ends = seq_lens.cumsum(dim=-1)
    total = ends[:, -1:]
    doc_ids = torch.searchsorted(ends.contiguous(), positions.contiguous(), right=True) + 1
    doc_ids = torch.where(positions < total, doc_ids, torch.zeros_like(doc_ids))

    same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
    not_padding = doc_ids > 0
    idx = torch.arange(seq_len, device=device)
    causal = idx.unsqueeze(0) <= idx.unsqueeze(1)
    allowed = same_doc & causal.unsqueeze(0) & not_padding.unsqueeze(2) & not_padding.unsqueeze(1)
    if sliding_window is not None and sliding_window > 0:
        allowed = allowed & ((idx.unsqueeze(1) - idx.unsqueeze(0)) < sliding_window).unsqueeze(0)

    min_value = torch.finfo(dtype).min if dtype.is_floating_point else -1e9
    return torch.where(
        allowed.unsqueeze(1),
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_value, dtype=dtype, device=device),
    )


def build_packed_causal_padding_mask_from_seq_ids(
    seq_ids: torch.Tensor,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Build a 4D packed causal mask from explicit packed sample ids.

    This is safer than reconstructing boundaries from ``seq_lens`` when the
    packer inserts per-sample padding for CP/compressor alignment.
    """
    if seq_ids.dim() == 1:
        seq_ids = seq_ids.unsqueeze(0)
    seq_ids = seq_ids.to(device=device, dtype=torch.long)
    if seq_ids.shape[1] != seq_len:
        raise ValueError(f"seq_ids must have sequence length {seq_len}, got {seq_ids.shape[1]}")

    idx = torch.arange(seq_len, device=device)
    same_doc = seq_ids.unsqueeze(2) == seq_ids.unsqueeze(1)
    not_padding = seq_ids >= 0
    causal = idx.unsqueeze(0) <= idx.unsqueeze(1)
    allowed = same_doc & causal.unsqueeze(0) & not_padding.unsqueeze(2) & not_padding.unsqueeze(1)
    if sliding_window is not None and sliding_window > 0:
        allowed = allowed & ((idx.unsqueeze(1) - idx.unsqueeze(0)) < sliding_window).unsqueeze(0)

    min_value = torch.finfo(dtype).min if dtype.is_floating_point else -1e9
    return torch.where(
        allowed.unsqueeze(1),
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_value, dtype=dtype, device=device),
    )


def eager_attention_with_sink(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager attention with per-head sink: appends an extra softmax column
    whose logit is ``module.sinks[h]`` and whose value-slot is zero.  Ported
    verbatim from HF PR 45616.
    """
    del kwargs
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : attn_weights.shape[-1]]
    sinks_weight = module.sinks_param(query) if hasattr(module, "sinks_param") else module.sinks
    sinks = sinks_weight.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined = torch.cat([attn_weights, sinks.to(attn_weights.dtype)], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1, dtype=torch.float32)[..., :-1]
    probs = F.dropout(probs, p=dropout, training=module.training).to(value_states.dtype)
    return torch.matmul(probs, value_states).transpose(1, 2).contiguous(), probs


class DeepseekV4FP32Parameter(nn.Module):
    """Callable holder for fp32 tensors that need their own FSDP unit."""

    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(value.to(torch.float32))

    def forward(self, anchor: torch.Tensor | None = None) -> torch.Tensor:
        # FSDP2 root pre-forward assumes at least one positional input.  These
        # parameter holders are otherwise argless, so callers pass an activation
        # anchor to keep checkpoint recompute + FSDP hooks on the supported path.
        del anchor
        return self.weight


class DeepseekV4Indexer(nn.Module):
    """HF PR 45616 port.  Picks the top-k compressed positions per query when
    ``compress_ratio == 4``.  Owns its own pool at ``index_head_dim`` plus a
    query projection + weights_proj head-mixer.
    """

    def __init__(self, config: DeepseekV4Config, backend: BackendConfig | None = None):
        super().__init__()
        self.backend = backend or BackendConfig()
        self.compress_ratio = 4
        # Indexer's pool is always at compress_ratio==4, which means overlap mode
        # (matching the released checkpoint's ``indexer.compressor.{ape,wkv,wgate}``
        # shapes of ``[ratio, 2*index_head_dim]`` / ``[2*index_head_dim, hidden_size]``).
        self.overlap = True
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.topk_query_block_size = int(getattr(config, "indexer_topk_query_block_size", 256) or 256)
        self.topk_key_block_size = int(getattr(config, "indexer_topk_key_block_size", 2048) or 2048)
        self.softmax_scale = self.head_dim**-0.5
        proj_dim = 2 * self.head_dim  # overlap mode
        self.wkv = nn.Linear(config.hidden_size, proj_dim, bias=False, dtype=torch.float32)
        self.wgate = nn.Linear(config.hidden_size, proj_dim, bias=False, dtype=torch.float32)
        self.ape_param = DeepseekV4FP32Parameter(torch.zeros(self.compress_ratio, proj_dim, dtype=torch.float32))
        self.kv_norm = initialize_rms_norm_module("torch_fp32", self.head_dim, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)

    @property
    def ape(self) -> torch.Tensor:
        return self.ape_param()

    def pool_projected(
        self,
        kv: torch.Tensor,
        gate: torch.Tensor,
        rotary: nn.Module,
        *,
        start_pos: int,
        pool_position_ids: torch.Tensor | None = None,
        pool_seq_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _pool_projected_windows(
            kv,
            gate,
            self.ape_param(kv),
            self.kv_norm,
            rotary,
            ratio=self.compress_ratio,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            overlap=self.overlap,
            start_pos=start_pos,
            pool_position_ids=pool_position_ids,
            pool_seq_ids=pool_seq_ids,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        rotary: nn.Module,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: DeepseekV4TrainCache,
        layer_idx: int,
        start_pos: int,
        query_hidden_states: torch.Tensor | None = None,
        query_q_residual: torch.Tensor | None = None,
        query_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        query_start: int = 0,
        query_positions: torch.Tensor | None = None,
        streaming_topk: bool = False,
        projected_kv: torch.Tensor | None = None,
        projected_gate: torch.Tensor | None = None,
        precomputed_pooled_kv: torch.Tensor | None = None,
        pool_position_ids: torch.Tensor | None = None,
        pool_seq_ids: torch.Tensor | None = None,
        query_seq_ids: torch.Tensor | None = None,
        pooled_seq_ids: torch.Tensor | None = None,
        query_local_positions: torch.Tensor | None = None,
        pooled_local_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _dsv4_debug(
            "indexer.forward.begin",
            layer_idx,
            hidden_states=hidden_states,
            query_hidden_states=query_hidden_states,
            projected_kv=projected_kv,
            projected_gate=projected_gate,
            precomputed_pooled_kv=precomputed_pooled_kv,
            query_start=query_start,
            query_positions=query_positions,
            streaming_topk=streaming_topk,
        )
        if precomputed_pooled_kv is None:
            hidden_states_fp32 = hidden_states.float()
            kv = self.wkv(hidden_states_fp32) if projected_kv is None else projected_kv
            gate = self.wgate(hidden_states_fp32) if projected_gate is None else projected_gate
            if kv.shape != gate.shape:
                raise ValueError(f"DeepSeek V4 indexer projected kv/gate shape mismatch: {kv.shape} vs {gate.shape}")
            _dsv4_debug("indexer.pool.before", layer_idx, kv=kv, gate=gate)
            batch = kv.shape[0]
            ready_kv, ready_gate, pool_base = cache.accumulate_windows(
                kv, gate, layer_idx, "indexer_state", self.compress_ratio, start_pos
            )
            new_pooled = self.pool_projected(
                ready_kv,
                ready_gate,
                rotary,
                start_pos=pool_base,
                pool_position_ids=pool_position_ids,
                pool_seq_ids=pool_seq_ids,
            )
            pooled_kv = cache.update_pool(new_pooled, layer_idx, "indexer_state")
            _dsv4_debug(
                "indexer.pool.after",
                layer_idx,
                ready_kv=ready_kv,
                ready_gate=ready_gate,
                pooled_kv=pooled_kv,
                pool_base=pool_base,
            )
        else:
            pooled_kv = precomputed_pooled_kv
            batch = pooled_kv.shape[0]
            _dsv4_debug("indexer.pool.precomputed", layer_idx, pooled_kv=pooled_kv)

        query_hidden_states = hidden_states if query_hidden_states is None else query_hidden_states
        query_q_residual = q_residual if query_q_residual is None else query_q_residual
        query_position_embeddings = (
            position_embeddings if query_position_embeddings is None else query_position_embeddings
        )
        if query_q_residual is None:
            raise ValueError("DeepSeek V4 indexer requires q_residual for query scoring")

        query_len = query_hidden_states.shape[1]
        cos, sin = query_position_embeddings
        q = self.wq_b(query_q_residual).view(batch, query_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = _apply_partial_rope(q, cos, sin, self.rope_head_dim).transpose(1, 2)
        weights = self.weights_proj(query_hidden_states).float() * (self.n_heads**-0.5)
        topk = min(self.index_topk, pooled_kv.shape[1])
        if topk <= 0:
            return torch.empty(batch, query_len, 0, dtype=torch.int32, device=pooled_kv.device)
        _dsv4_debug(
            "indexer.topk.before",
            layer_idx,
            q=q,
            pooled_kv=pooled_kv,
            weights=weights,
            topk=topk,
            query_len=query_len,
            query_start=query_start,
        )
        dense_score_elements = batch * query_len * pooled_kv.shape[1]
        try:
            streaming_threshold = int(os.environ.get("DSV4_CP_INDEXER_STREAMING_THRESHOLD", "16777216"))
        except ValueError:
            streaming_threshold = 16777216
        query_positions_contiguous = _query_positions_are_contiguous(query_positions, query_start, query_len)
        force_dense_topk = _dsv4_env_flag("DSV4_CP_INDEXER_DENSE", "0") and query_positions_contiguous
        use_streaming_topk = not force_dense_topk and (
            _dsv4_env_flag("DSV4_CP_INDEXER_STREAMING", "0")
            or streaming_topk
            or query_len != hidden_states.shape[1]
            or query_start != 0
            or not query_positions_contiguous
            or dense_score_elements > streaming_threshold
        )
        if use_streaming_topk:
            with torch.no_grad():
                result = streaming_indexer_topk_indices_torch(
                    q,
                    pooled_kv,
                    weights,
                    topk=topk,
                    compress_ratio=self.compress_ratio,
                    softmax_scale=self.softmax_scale,
                    query_start=query_start,
                    query_positions=query_positions,
                    query_local_positions=query_local_positions,
                    query_seq_ids=query_seq_ids,
                    pooled_seq_ids=pooled_seq_ids,
                    pooled_local_positions=pooled_local_positions,
                    query_block_size=self.topk_query_block_size,
                    key_block_size=self.topk_key_block_size,
                )
            _dsv4_debug(
                "indexer.streaming_topk.after",
                layer_idx,
                result=result,
                dense_score_elements=dense_score_elements,
                streaming_threshold=streaming_threshold,
            )
            return result
        _dsv4_debug(
            "indexer.scores.before",
            layer_idx,
            q=q,
            pooled_kv=pooled_kv,
            weights=weights,
            dense_topk_for_cp=streaming_topk,
            query_start=query_start,
        )
        with torch.no_grad():
            index_scores = dsv4_indexer_scores(
                q,
                pooled_kv,
                weights,
                compress_ratio=self.compress_ratio,
                softmax_scale=self.softmax_scale,
                backend=_dsv4_kernel_backend(self.backend),
                query_start=query_start,
                query_positions=query_positions,
                query_local_positions=query_local_positions,
                query_seq_ids=query_seq_ids,
                pooled_seq_ids=pooled_seq_ids,
                pooled_local_positions=pooled_local_positions,
            )
            result = index_scores.topk(topk, dim=-1).indices.to(torch.int32)
        _dsv4_debug("indexer.scores.after", layer_idx, index_scores=index_scores, result=result)
        return result


class DeepseekV4Compressor(nn.Module):
    """HF PR 45616 port.  Long-range KV branch.  Pools ``compress_ratio`` tokens
    into one compressed KV; when ``ratio == 4`` the Indexer narrows the pool.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        compress_ratio: int,
        head_dim: int,
        backend: BackendConfig | None = None,
    ):
        super().__init__()
        self.backend = backend or BackendConfig()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        # Overlap mode (compress_ratio==4) doubles the feature dim of wkv / wgate /
        # ape — the released DSV4-Flash checkpoint was trained that way to give each
        # compressed token cross-window context. Non-overlap mode (compress_ratio==128)
        # keeps a flat head_dim. ``kv_norm`` always normalizes over ``head_dim`` because
        # the overlap_transform inside ``_pool_windows`` collapses 2*head_dim → head_dim
        # before the norm runs.
        self.overlap = compress_ratio == 4
        coff = 2 if self.overlap else 1
        proj_dim = coff * head_dim
        self.wkv = nn.Linear(config.hidden_size, proj_dim, bias=False, dtype=torch.float32)
        self.wgate = nn.Linear(config.hidden_size, proj_dim, bias=False, dtype=torch.float32)
        self.ape_param = DeepseekV4FP32Parameter(torch.zeros(compress_ratio, proj_dim, dtype=torch.float32))
        self.kv_norm = initialize_rms_norm_module("torch_fp32", head_dim, eps=config.rms_norm_eps)
        self.indexer: DeepseekV4Indexer | None = (
            DeepseekV4Indexer(config, backend=self.backend) if compress_ratio == 4 else None
        )
        self._hca_param_sync_group = None

    @property
    def ape(self) -> torch.Tensor:
        return self.ape_param()

    def _set_hca_param_sync_group(self, process_group) -> None:
        self._hca_param_sync_group = process_group

    def _compute_fsdp_group_has_complete_hca_window(
        self,
        local_has_complete_hca_window: bool,
        device: torch.device,
    ) -> bool:
        process_group = self._hca_param_sync_group
        if process_group is None or not dist.is_available() or not dist.is_initialized():
            return local_has_complete_hca_window
        if dist.get_world_size(group=process_group) == 1:
            return local_has_complete_hca_window

        flag = torch.tensor(int(local_has_complete_hca_window), device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=process_group)
        return bool(flag.item())

    def pool_projected(
        self,
        kv: torch.Tensor,
        gate: torch.Tensor,
        rotary: nn.Module,
        *,
        start_pos: int,
        pool_position_ids: torch.Tensor | None = None,
        pool_seq_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _pool_projected_windows(
            kv,
            gate,
            self.ape_param(kv),
            self.kv_norm,
            rotary,
            ratio=self.compress_ratio,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            overlap=self.overlap,
            start_pos=start_pos,
            pool_position_ids=pool_position_ids,
            pool_seq_ids=pool_seq_ids,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor | None,
        rotary: nn.Module,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: DeepseekV4TrainCache,
        layer_idx: int,
        start_pos: int,
        indexer_query_hidden_states: torch.Tensor | None = None,
        indexer_q_residual: torch.Tensor | None = None,
        indexer_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        indexer_query_start: int = 0,
        indexer_query_positions: torch.Tensor | None = None,
        streaming_indexer_topk: bool = False,
        projected_kv: torch.Tensor | None = None,
        projected_gate: torch.Tensor | None = None,
        indexer_projected_kv: torch.Tensor | None = None,
        indexer_projected_gate: torch.Tensor | None = None,
        precomputed_pooled: torch.Tensor | None = None,
        indexer_precomputed_pooled: torch.Tensor | None = None,
        pool_position_ids: torch.Tensor | None = None,
        pool_seq_ids: torch.Tensor | None = None,
        indexer_pool_position_ids: torch.Tensor | None = None,
        indexer_pool_seq_ids: torch.Tensor | None = None,
        indexer_query_seq_ids: torch.Tensor | None = None,
        indexer_pooled_seq_ids: torch.Tensor | None = None,
        indexer_query_local_positions: torch.Tensor | None = None,
        indexer_pooled_local_positions: torch.Tensor | None = None,
        enable_hca_fsdp_graph_alignment: bool = False,
    ) -> torch.Tensor:
        _dsv4_debug(
            "compressor.forward.begin",
            layer_idx,
            hidden_states=hidden_states,
            projected_kv=projected_kv,
            projected_gate=projected_gate,
            precomputed_pooled=precomputed_pooled,
            indexer_precomputed_pooled=indexer_precomputed_pooled,
            has_indexer=self.indexer is not None,
            indexer_query_start=indexer_query_start,
            indexer_query_positions=indexer_query_positions,
            streaming_indexer_topk=streaming_indexer_topk,
        )
        if precomputed_pooled is None:
            hidden_states_fp32 = hidden_states.float()
            kv = self.wkv(hidden_states_fp32) if projected_kv is None else projected_kv
            gate = self.wgate(hidden_states_fp32) if projected_gate is None else projected_gate
            if kv.shape != gate.shape:
                raise ValueError(f"DeepSeek V4 compressor projected kv/gate shape mismatch: {kv.shape} vs {gate.shape}")
            _dsv4_debug("compressor.pool.before", layer_idx, kv=kv, gate=gate)
            ready_kv, ready_gate, pool_base = cache.accumulate_windows(
                kv, gate, layer_idx, "compressor_state", self.compress_ratio, start_pos
            )
            local_has_complete_hca_window = ready_kv.shape[1] > 0
            fsdp_group_has_complete_hca_window = (
                self._compute_fsdp_group_has_complete_hca_window(local_has_complete_hca_window, kv.device)
                if enable_hca_fsdp_graph_alignment
                else local_has_complete_hca_window
            )
            needs_masked_synthetic_hca_window = (
                enable_hca_fsdp_graph_alignment
                and self.indexer is None
                and fsdp_group_has_complete_hca_window
                and not local_has_complete_hca_window
                and 0 < kv.shape[1] < self.compress_ratio
            )
            if needs_masked_synthetic_hca_window:
                # Mixed short/long FSDP2 groups must create identical HCA
                # compressor autograd edges. The synthetic pooled token is later
                # fully masked by the normal compressed-KV causal rule.
                pad_len = self.compress_ratio - kv.shape[1]
                pad_shape = (kv.shape[0], pad_len, kv.shape[-1])
                ready_kv = torch.cat([kv, kv.new_zeros(pad_shape)], dim=1)
                ready_gate = torch.cat([gate, gate.new_zeros(pad_shape)], dim=1)
                pool_position_ids = _pad_metadata_to_len(pool_position_ids, ready_kv.shape[1], value=0)
                pool_seq_ids = _pad_metadata_to_len(pool_seq_ids, ready_kv.shape[1], value=-1)
                pool_base = max(0, start_pos)
            new_pooled = self.pool_projected(
                ready_kv,
                ready_gate,
                rotary,
                start_pos=pool_base,
                pool_position_ids=pool_position_ids,
                pool_seq_ids=pool_seq_ids,
            )
            pooled = cache.update_pool(new_pooled, layer_idx, "compressor_state").unsqueeze(1)
            _dsv4_debug(
                "compressor.pool.after",
                layer_idx,
                ready_kv=ready_kv,
                ready_gate=ready_gate,
                pooled=pooled,
                pool_base=pool_base,
            )
        else:
            pooled = precomputed_pooled.unsqueeze(1)
            _dsv4_debug("compressor.pool.precomputed", layer_idx, pooled=pooled)

        # Indexer narrows the attended compressed positions per query.  The
        # caller (DSV4Attention) is responsible for turning ``indexer_topk``
        # into an additive attention mask; we do NOT pre-gather here.  The
        # earlier per-query ``torch.gather`` produced an
        # ``[B, 1, S*topk, D]`` tensor that, when concatenated to ``full_kv``
        # and run through dense attention with ``F.pad(value=0.0)``, let
        # every query attend to every other query's gathered slice — a
        # silent non-causal leak (verified empirically: layer 2 attention
        # output cosine-vs-reference jumps from 0.81 to 0.99+ once we move
        # to mask-driven sparse semantics).
        #
        # ``indexer_topk`` follows the reference contract from
        # ``dsv4flash/inference/model.py:472-475``: shape ``[B, S, K]`` with
        # entries that are either valid pool indices in ``[0, P_total)``
        # or ``-1`` for "do not attend" (masked by causality).
        indexer_topk: torch.LongTensor | None = None
        if self.indexer is not None:
            _dsv4_debug("compressor.indexer.before", layer_idx, pooled=pooled)
            raw_topk = self.indexer(
                hidden_states,
                q_residual,
                rotary,
                position_embeddings,
                cache,
                layer_idx,
                start_pos,
                query_hidden_states=indexer_query_hidden_states,
                query_q_residual=indexer_q_residual,
                query_position_embeddings=indexer_position_embeddings,
                query_start=indexer_query_start,
                query_positions=indexer_query_positions,
                streaming_topk=streaming_indexer_topk,
                projected_kv=indexer_projected_kv,
                projected_gate=indexer_projected_gate,
                precomputed_pooled_kv=indexer_precomputed_pooled,
                pool_position_ids=indexer_pool_position_ids,
                pool_seq_ids=indexer_pool_seq_ids,
                query_seq_ids=indexer_query_seq_ids,
                pooled_seq_ids=indexer_pooled_seq_ids,
                query_local_positions=indexer_query_local_positions,
                pooled_local_positions=indexer_pooled_local_positions,
            )
            query_len = raw_topk.shape[1]
            if indexer_query_positions is None:
                query_positions = torch.arange(
                    indexer_query_start + 1,
                    indexer_query_start + query_len + 1,
                    device=raw_topk.device,
                )
                threshold = (query_positions // self.compress_ratio).unsqueeze(1)
            else:
                query_positions = indexer_query_positions.to(device=raw_topk.device, dtype=torch.int64)
                if query_positions.numel() != query_len:
                    raise ValueError(
                        "indexer_query_positions length must match indexer query length "
                        f"(got {query_positions.numel()} vs {query_len})"
                    )
                threshold = ((query_positions + 1) // self.compress_ratio).unsqueeze(1)
            if indexer_pooled_seq_ids is not None:
                pooled_seq_ids = indexer_pooled_seq_ids.to(device=raw_topk.device, dtype=torch.int64)
                if indexer_query_seq_ids is not None:
                    if indexer_query_local_positions is None:
                        query_local_positions = query_seq_positions_from_ids(
                            indexer_query_seq_ids.to(device=raw_topk.device, dtype=torch.int64)
                        )
                    else:
                        query_local_positions = indexer_query_local_positions.to(device=raw_topk.device, dtype=torch.int64)
                        if query_local_positions.shape != indexer_query_seq_ids.shape:
                            raise ValueError(
                                "indexer_query_local_positions must match indexer_query_seq_ids shape "
                                f"({tuple(query_local_positions.shape)} vs {tuple(indexer_query_seq_ids.shape)})"
                            )
                    threshold = ((query_local_positions + 1) // self.compress_ratio).unsqueeze(-1)
                if indexer_pooled_local_positions is None:
                    pooled_positions = pooled_seq_positions_from_ids(pooled_seq_ids)
                else:
                    pooled_positions = indexer_pooled_local_positions.to(device=raw_topk.device, dtype=torch.int64)
                    if pooled_positions.shape != pooled_seq_ids.shape:
                        raise ValueError(
                            "indexer_pooled_local_positions must match indexer_pooled_seq_ids shape "
                            f"({tuple(pooled_positions.shape)} vs {tuple(pooled_seq_ids.shape)})"
                        )
                safe_topk = raw_topk.clamp(min=0, max=max(pooled_positions.shape[1] - 1, 0)).to(torch.int64)
                topk_seq_ids = torch.gather(
                    pooled_seq_ids.unsqueeze(1).expand(-1, query_len, -1),
                    dim=-1,
                    index=safe_topk,
                )
                topk_positions = torch.gather(
                    pooled_positions.unsqueeze(1).expand(-1, query_len, -1),
                    dim=-1,
                    index=safe_topk,
                )
                causal_invalid = (raw_topk < 0) | (topk_positions < 0) | (topk_positions >= threshold)
                if indexer_query_seq_ids is not None:
                    query_seq_ids = indexer_query_seq_ids.to(device=raw_topk.device, dtype=torch.int64)
                    causal_invalid = causal_invalid | (topk_seq_ids != query_seq_ids.unsqueeze(-1)) | (
                        query_seq_ids.unsqueeze(-1) < 0
                    )
            else:
                causal_invalid = raw_topk >= threshold
            indexer_topk = torch.where(causal_invalid, torch.full_like(raw_topk, -1), raw_topk)
            _dsv4_debug("compressor.indexer.after", layer_idx, raw_topk=raw_topk, indexer_topk=indexer_topk)
        _dsv4_debug("compressor.forward.end", layer_idx, pooled=pooled, indexer_topk=indexer_topk)
        return pooled, indexer_topk


# ---------------------------------------------------------------------------
# HC (Hyper-Connections) — ported verbatim from HuggingFace transformers
# PR 45616 (Arthur Zucker, "Add DeepSeek V4").  Source-of-truth reference:
# ``transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py``
# classes ``DeepseekV4HyperConnection`` (lines 613-670) and
# ``DeepseekV4HyperHead`` (lines 673-690).
#
# The previous pure-torch port (mean-pool / softmax-comb) had three silent
# divergences vs HF: (a) ``comb`` used row-softmax where HF uses ``sigmoid``,
# (b) ``post`` had a ``2 *`` prefactor + missing ``+eps``, (c) the mixer was
# wrapped in ``torch.no_grad()``.  Rather than patch those line-by-line,
# swap wholesale to the HF classes so future HF updates flow through cleanly
# via the adapter's state-dict rename rules.
# ---------------------------------------------------------------------------


class DeepseekV4HyperConnection(nn.Module):
    """Per-site HyperConnection mixer (attention or FFN).  Ported from
    ``transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py``
    class ``DeepseekV4HyperConnection``.

    Owns ``fn`` (packed linear), ``base`` (bias), and ``scale`` (scalar
    per-head gains).  ``compute_weights`` produces three mixer tensors:

      - ``pre``   [B, S, H]       : sigmoid-gated collapse weights
      - ``post``  [B, S, H]       : sigmoid-gated expand weights
      - ``comb``  [B, S, H, H]    : doubly-stochastic combination matrix
                                    from Sinkhorn-normalising sigmoid gates

    All math runs in fp32 regardless of the outer cast policy; parameters
    cast themselves via ``.float()`` on each forward.  HF lists these params
    in ``_keep_in_fp32_modules_strict`` — the KAutomodel adapter does the
    same via submodule-name matching.
    """

    def __init__(
        self,
        hc_mult: int,
        hidden_size: int,
        hc_sinkhorn_iters: int,
        hc_eps: float,
        rms_norm_eps: float,
        sinkhorn_backend: str = "torch",
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.norm_eps = rms_norm_eps
        self.sinkhorn_backend = sinkhorn_backend
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))

    def compute_weights(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_streams.flatten(start_dim=2).float()  # [B, S, H*D]
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        # HC mixer params are kept in fp32 for Sinkhorn stability — cast defensively.
        mix = torch.nn.functional.linear(flat, self.fn.float()) * rsqrt  # [B, S, (2+H)*H]
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)
        hc = self.hc_mult

        # ``pre`` and ``post`` have DIFFERENT formulas in the released DSV4-Flash
        # (see ``dsv4flash/inference/kernel.py:hc_split_sinkhorn_kernel`` 391-394):
        #   pre  = sigmoid(...) + eps     range (eps, 1+eps]
        #   post = 2 * sigmoid(...)       range (0, 2)  — NO +eps, AND a 2x prefactor
        # HF transformers PR 45616 / 45643 treats post identically to pre (sigmoid
        # + eps), which makes ``post`` half the magnitude the released weights
        # were trained against — verified empirically on the parity test
        # (auto post std = 0.5x ref post std before this fix).
        pre = torch.sigmoid(mix[..., :hc] * pre_scale + self.base[:hc].float()) + self.hc_eps
        post = 2.0 * torch.sigmoid(mix[..., hc : 2 * hc] * post_scale + self.base[hc : 2 * hc].float())

        # ``comb`` uses softmax(dim=-1) on raw logits + eps, then sinkhorn.  HF
        # uses sigmoid + eps + sinkhorn — also a divergence from the reference
        # kernel.  Reference (kernel.py:395-413):
        #   1. comb_logit = mix * scale + base
        #   2. row_softmax(dim=-1) + eps   (numerically stable, NOT sigmoid)
        #   3. col-norm / sum(dim=-2)
        #   4. for sinkhorn_iters - 1: row-norm / sum(dim=-1) ; col-norm / sum(dim=-2)
        comb_logit = (
            mix[..., 2 * hc :].view(*mix.shape[:-1], hc, hc) * comb_scale + self.base[2 * hc :].view(hc, hc).float()
        )
        comb = dsv4_sinkhorn_normalize(
            comb_logit,
            backend=self.sinkhorn_backend,
            repeat=self.hc_sinkhorn_iters,
            eps=self.hc_eps,
        )
        return pre, post, comb

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.compute_weights(hidden_streams)


class DeepseekV4HyperHead(nn.Module):
    """Final HC-stream collapse before the shared RMSNorm + ``lm_head``.
    Ported from ``modular_deepseek_v4.py`` class ``DeepseekV4HyperHead``.

    Sigmoid-weighted sum over the ``hc_mult`` streams (no Sinkhorn).  Used
    once at the end of ``DeepseekV4Model.forward`` to go from
    ``[B, S, H, D]`` back to ``[B, S, D]``.
    """

    def __init__(self, hc_mult: int, hidden_size: int, hc_eps: float, rms_norm_eps: float):
        super().__init__()
        self.hc_mult = hc_mult
        self.norm_eps = rms_norm_eps
        self.eps = hc_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = torch.nn.functional.linear(flat, self.hc_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return _dsv4_hc_collapse(pre, x)


# ---------------------------------------------------------------------------
# DeepseekV4Attention — port of HF PR 45616's DeepseekV4Attention with:
#   - no inheritance from DeepseekV3Attention (we don't want the HF
#     PreTrainedModel scaffolding);
#   - ``position_embeddings`` passed in from DeepseekV4Model as a ``(cos, sin)``
#     tuple produced by a matching ``DeepseekV4RotaryEmbedding`` (plus a
#     separate rotary / position_embeddings pair for the compressor path);
#   - ``past_key_values`` always ``None`` on the training path; the compressor
#     / indexer use a per-forward ``DeepseekV4TrainCache`` shim that behaves
#     the same as HF's ``DeepseekV4Cache`` within a single call;
#   - explicit backend dispatch for SDPA, the dense torch reference, and the
#     optional DeepSeek V4 sparse-attention kernels.
# ---------------------------------------------------------------------------


class DeepseekV4Attention(nn.Module):
    """Sliding-window attention + Compressor + Indexer + attention sink.

    Single-head KV (``num_key_value_heads=1``), grouped low-rank output via
    :class:`DeepseekV4GroupedLinear`.  ``compress_ratio == 0`` layers skip
    the compressor / indexer and run pure SWA.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int, backend: BackendConfig | None = None):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        self.layer_idx = layer_idx
        self.compress_ratio = int(config.compress_ratios[layer_idx]) if config.compress_ratios else 0
        self.num_heads = config.num_attention_heads
        # Single KV head broadcast to all attention heads (``num_key_value_groups == num_heads``).
        self.num_key_value_groups = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.sliding_window = int(getattr(config, "sliding_window", 128) or 128)
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0) or 0.0)
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = initialize_rms_norm_module("torch_fp32", config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = initialize_rms_norm_module("torch_fp32", self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks_param = DeepseekV4FP32Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self._cp_mesh = None

        self.compressor = (
            DeepseekV4Compressor(config, self.compress_ratio, self.head_dim, backend=self.backend)
            if self.compress_ratio
            else None
        )

    @property
    def sinks(self) -> torch.Tensor:
        return self.sinks_param()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_embeddings_compress: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_compress: nn.Module | None = None,
        position_ids: torch.Tensor | None = None,
        start_pos: int = 0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cp_layout = _dsv4_cp_layout(kwargs.get("dsv4_cp_layout", None))
        batch, seq_len = hidden_states.shape[:2]
        attn_backend = _dsv4_kernel_backend(self.backend)
        cp_mesh = getattr(self, "_cp_mesh", None)
        cp_enabled = _cp_mesh_enabled(cp_mesh)
        if cp_enabled and attn_backend != "tilelang":
            raise RuntimeError("DeepSeek V4 manual CP requires backend.attn='tilelang'.")
        cp_zigzag = cp_enabled and _cp_layout_is_zigzag(cp_layout)
        packed_seq_ids = kwargs.get("dsv4_seq_ids", kwargs.get("seq_ids", kwargs.get("_packed_seq_ids", None)))
        token_positions = kwargs.get("dsv4_token_positions", None)
        packed_sequence = isinstance(packed_seq_ids, torch.Tensor)
        if token_positions is not None and not isinstance(token_positions, torch.Tensor):
            token_positions = None
        if cp_enabled and packed_sequence:
            if token_positions is None:
                raise ValueError("DeepSeek V4 packed manual CP requires dsv4_token_positions")
            if token_positions.shape != packed_seq_ids.shape:
                raise ValueError(
                    "DeepSeek V4 packed manual CP requires dsv4_token_positions to match packed seq_ids shape "
                    f"({tuple(token_positions.shape)} vs {tuple(packed_seq_ids.shape)})"
                )
        _dsv4_debug(
            "attn.forward.begin",
            self.layer_idx,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            packed_seq_ids=packed_seq_ids,
            token_positions=token_positions,
            seq_len=seq_len,
            compress_ratio=self.compress_ratio,
            sliding_window=self.sliding_window,
            attn_backend=attn_backend,
            cp_enabled=cp_enabled,
            cp_layout=cp_layout,
            cp_size=cp_mesh.size() if cp_enabled else 1,
            cp_rank=_cp_mesh_rank(cp_mesh) if cp_enabled else 0,
        )
        # IMPORTANT: for compress_ratio>0 layers the released DSV4-Flash uses
        # the compress-rope (theta=160000 + YaRN) for the MAIN attention Q/KV
        # too, NOT just for the compressor sub-module.  Reference at
        # ``dsv4flash/inference/model.py:476-501`` builds ``self.freqs_cis``
        # with ``compress_rope_theta`` whenever ``compress_ratio != 0``.  The
        # caller passes both ``position_embeddings`` (theta=10000, no YaRN)
        # and ``position_embeddings_compress`` (theta=160000, YaRN); we pick
        # the right one here based on compress_ratio.
        if self.compress_ratio and position_embeddings_compress is not None:
            cos, sin = position_embeddings_compress
        else:
            cos, sin = position_embeddings

        q_residual = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(q_residual).view(batch, seq_len, self.num_heads, self.head_dim)
        kv = self.kv_norm(self.wkv(hidden_states)).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)

        # Per-head, non-learnable rsqrt on Q before RoPE (matches reference
        # ``dsv4flash/inference/model.py:498``; missing from HF PR 45616).
        q = _chunked_q_rms_norm_(q, self.config.rms_norm_eps).transpose(1, 2)

        q = _apply_partial_rope(q, cos, sin, self.rope_head_dim)
        kv = _apply_partial_rope(kv, cos, sin, self.rope_head_dim)
        _dsv4_debug("attn.qkv.after_rope", self.layer_idx, q=q, kv=kv, q_residual=q_residual)

        raw_key_start = 0
        query_positions = None
        query_key_positions = None
        query_seq_ids = packed_seq_ids if packed_sequence else None
        query_local_position_ids = position_ids if packed_sequence and isinstance(position_ids, torch.Tensor) else None
        raw_key_seq_ids = None
        full_packed_seq_ids = None
        full_pool_position_ids = None
        if cp_enabled:
            # Avoid reading CUDA position_ids with .item() under CP: the shard
            # start is determined by the CP layout, and .item() would force a
            # full CUDA stream sync before the CP collectives. Zigzag CP uses
            # per-token global positions because each rank owns two disjoint
            # global chunks.
            cp_rank = _cp_mesh_rank(cp_mesh)
            query_start = cp_rank * (seq_len // 2 if cp_zigzag else seq_len)
            query_positions = _query_positions_1d(position_ids, seq_len, kv.device, query_start)
            query_token_positions = (
                _query_positions_1d(token_positions, seq_len, kv.device, query_start)
                if token_positions is not None
                else None
            )
            _dsv4_debug(
                "attn.raw_kv.gather.before",
                self.layer_idx,
                kv=kv,
                query_start=query_start,
                query_positions=query_positions,
                query_token_positions=query_token_positions,
                cp_layout=cp_layout,
            )
            full_kv, raw_key_start = _cp_gather_sliding_window_kv(
                kv,
                cp_mesh,
                self.sliding_window,
                cp_layout=cp_layout,
                debug_layer_idx=self.layer_idx,
            )
            if packed_sequence:
                raw_key_seq_ids, _ = _cp_gather_sliding_window_metadata(
                    packed_seq_ids,
                    cp_mesh,
                    self.sliding_window,
                    cp_layout=cp_layout,
                )
                full_packed_seq_ids = _gather_full_cp_metadata(packed_seq_ids, cp_mesh, cp_layout=cp_layout)
                full_pool_position_ids = _gather_full_cp_metadata(position_ids, cp_mesh, cp_layout=cp_layout)
            query_key_positions = (query_token_positions if query_token_positions is not None else query_positions) - raw_key_start
            _dsv4_debug(
                "attn.raw_kv.gather.after",
                self.layer_idx,
                full_kv=full_kv,
                raw_key_seq_ids=raw_key_seq_ids,
                full_packed_seq_ids=full_packed_seq_ids,
                full_pool_position_ids=full_pool_position_ids,
                raw_key_start=raw_key_start,
                query_start=query_start,
                query_key_positions=query_key_positions,
            )
        else:
            query_start = _first_position(position_ids, 0)
            full_kv = kv
            if packed_sequence:
                raw_key_seq_ids = packed_seq_ids
                full_packed_seq_ids = packed_seq_ids
                full_pool_position_ids = position_ids
                if token_positions is not None:
                    query_key_positions = _query_positions_1d(token_positions, seq_len, kv.device, query_start)
                elif packed_sequence:
                    query_key_positions = torch.arange(seq_len, device=kv.device, dtype=torch.int64)
        if packed_sequence and self.compress_ratio > 0:
            _validate_packed_pool_alignment(full_packed_seq_ids, int(self.compress_ratio))
        n_pooled = 0
        indexer_topk: torch.LongTensor | None = None
        compressor_pooled_seq_ids = None
        compressor_pooled_positions = None

        if self.compressor is not None:
            assert rotary_compress is not None and position_embeddings_compress is not None, (
                "DeepseekV4Attention: compressor enabled but no rotary_compress / "
                "position_embeddings_compress supplied by the Block/Model."
            )
            compressor_projected_kv = None
            compressor_projected_gate = None
            indexer_projected_kv = None
            indexer_projected_gate = None
            compressor_precomputed_pooled = None
            indexer_precomputed_pooled = None
            compressor_pool_position_ids = full_pool_position_ids if packed_sequence else None
            compressor_pool_seq_ids = full_packed_seq_ids if packed_sequence else None
            indexer_pool_position_ids = full_pool_position_ids if packed_sequence else None
            indexer_pool_seq_ids = full_packed_seq_ids if packed_sequence else None
            indexer_pooled_seq_ids = (
                _pool_seq_ids(
                    full_packed_seq_ids,
                    self.compressor.indexer.compress_ratio,
                    overlap=self.compressor.indexer.overlap,
                )
                if packed_sequence and self.compressor.indexer is not None
                else None
            )
            indexer_pooled_positions = (
                _pool_position_ids(
                    full_pool_position_ids,
                    self.compressor.indexer.compress_ratio,
                )
                if packed_sequence and self.compressor.indexer is not None
                else None
            )
            if cp_enabled:
                compressor_hidden_states = hidden_states
                if not packed_sequence:
                    compressor_precomputed_pooled = _cp_pool_projected_and_gather(
                        hidden_states,
                        self.compressor,
                        rotary_compress,
                        cp_mesh,
                        query_start,
                        cp_layout=cp_layout,
                        debug_layer_idx=self.layer_idx,
                        debug_label="attn.compressor.precompute",
                    )
                if compressor_precomputed_pooled is None:
                    _dsv4_debug("attn.compressor.fallback_project.before", self.layer_idx, hidden_states=hidden_states)
                    hidden_states_fp32 = hidden_states.float()
                    compressor_local_kv = self.compressor.wkv(hidden_states_fp32)
                    compressor_local_gate = self.compressor.wgate(hidden_states_fp32)
                    _dsv4_debug(
                        "attn.compressor.fallback_project.after",
                        self.layer_idx,
                        compressor_local_kv=compressor_local_kv,
                        compressor_local_gate=compressor_local_gate,
                    )
                    if cp_zigzag:
                        compressor_projected_kv = _cp_all_gather_zigzag_halves(
                            compressor_local_kv,
                            cp_mesh,
                            dim=1,
                            debug_layer_idx=self.layer_idx,
                            debug_label="attn.compressor.fallback_kv",
                        )
                    else:
                        compressor_projected_kv = _cp_all_gather(
                            compressor_local_kv,
                            cp_mesh,
                            dim=1,
                            debug_layer_idx=self.layer_idx,
                            debug_label="attn.compressor.fallback_kv",
                        )
                    if cp_zigzag:
                        compressor_projected_gate = _cp_all_gather_zigzag_halves(
                            compressor_local_gate,
                            cp_mesh,
                            dim=1,
                            debug_layer_idx=self.layer_idx,
                            debug_label="attn.compressor.fallback_gate",
                        )
                    else:
                        compressor_projected_gate = _cp_all_gather(
                            compressor_local_gate,
                            cp_mesh,
                            dim=1,
                            debug_layer_idx=self.layer_idx,
                            debug_label="attn.compressor.fallback_gate",
                        )
                if self.compressor.indexer is not None:
                    if not packed_sequence:
                        indexer_precomputed_pooled = _cp_pool_projected_and_gather(
                            hidden_states,
                            self.compressor.indexer,
                            rotary_compress,
                            cp_mesh,
                            query_start,
                            cp_layout=cp_layout,
                            debug_layer_idx=self.layer_idx,
                            debug_label="attn.indexer.precompute",
                        )
                    if indexer_precomputed_pooled is None:
                        _dsv4_debug("attn.indexer.fallback_project.before", self.layer_idx, hidden_states=hidden_states)
                        hidden_states_fp32 = hidden_states.float()
                        indexer_local_kv = self.compressor.indexer.wkv(hidden_states_fp32)
                        indexer_local_gate = self.compressor.indexer.wgate(hidden_states_fp32)
                        _dsv4_debug(
                            "attn.indexer.fallback_project.after",
                            self.layer_idx,
                            indexer_local_kv=indexer_local_kv,
                            indexer_local_gate=indexer_local_gate,
                        )
                        if cp_zigzag:
                            indexer_projected_kv = _cp_all_gather_zigzag_halves(
                                indexer_local_kv,
                                cp_mesh,
                                dim=1,
                                debug_layer_idx=self.layer_idx,
                                debug_label="attn.indexer.fallback_kv",
                            )
                            indexer_projected_gate = _cp_all_gather_zigzag_halves(
                                indexer_local_gate,
                                cp_mesh,
                                dim=1,
                                debug_layer_idx=self.layer_idx,
                                debug_label="attn.indexer.fallback_gate",
                            )
                        else:
                            indexer_projected_kv = _cp_all_gather(
                                indexer_local_kv,
                                cp_mesh,
                                dim=1,
                                debug_layer_idx=self.layer_idx,
                                debug_label="attn.indexer.fallback_kv",
                            )
                            indexer_projected_gate = _cp_all_gather(
                                indexer_local_gate,
                                cp_mesh,
                                dim=1,
                                debug_layer_idx=self.layer_idx,
                                debug_label="attn.indexer.fallback_gate",
                            )
                position_embeddings_compress_for_compressor = position_embeddings_compress
                indexer_hidden_states = hidden_states
                indexer_q_residual = q_residual
                indexer_position_embeddings = position_embeddings_compress
            else:
                compressor_hidden_states = hidden_states
                position_embeddings_compress_for_compressor = position_embeddings_compress
                indexer_hidden_states = hidden_states
                indexer_q_residual = q_residual
                indexer_position_embeddings = position_embeddings_compress

            cache = DeepseekV4TrainCache()
            _dsv4_debug(
                "attn.compressor.forward.before",
                self.layer_idx,
                compressor_precomputed_pooled=compressor_precomputed_pooled,
                indexer_precomputed_pooled=indexer_precomputed_pooled,
                compressor_projected_kv=compressor_projected_kv,
                compressor_projected_gate=compressor_projected_gate,
                indexer_projected_kv=indexer_projected_kv,
                indexer_projected_gate=indexer_projected_gate,
            )
            pooled, indexer_topk = self.compressor(
                compressor_hidden_states,
                q_residual=q_residual,
                rotary=rotary_compress,
                position_embeddings=position_embeddings_compress_for_compressor,
                cache=cache,
                layer_idx=self.layer_idx,
                start_pos=start_pos,
                indexer_query_hidden_states=indexer_hidden_states,
                indexer_q_residual=indexer_q_residual,
                indexer_position_embeddings=indexer_position_embeddings,
                indexer_query_start=query_start,
                indexer_query_positions=query_positions,
                streaming_indexer_topk=cp_enabled,
                projected_kv=compressor_projected_kv,
                projected_gate=compressor_projected_gate,
                indexer_projected_kv=indexer_projected_kv,
                indexer_projected_gate=indexer_projected_gate,
                precomputed_pooled=compressor_precomputed_pooled,
                indexer_precomputed_pooled=indexer_precomputed_pooled,
                pool_position_ids=compressor_pool_position_ids if compressor_precomputed_pooled is None else None,
                pool_seq_ids=compressor_pool_seq_ids if compressor_precomputed_pooled is None else None,
                indexer_pool_position_ids=indexer_pool_position_ids if indexer_precomputed_pooled is None else None,
                indexer_pool_seq_ids=indexer_pool_seq_ids if indexer_precomputed_pooled is None else None,
                indexer_query_seq_ids=query_seq_ids,
                indexer_pooled_seq_ids=indexer_pooled_seq_ids,
                indexer_query_local_positions=query_local_position_ids,
                indexer_pooled_local_positions=indexer_pooled_positions,
                enable_hca_fsdp_graph_alignment=(
                    self.training
                    and self.compress_ratio == 128
                    and (attention_mask is not None or cp_enabled or packed_sequence)
                ),
            )
            n_pooled = pooled.shape[2]
            compressor_pooled_seq_ids = _pool_seq_ids(
                full_packed_seq_ids,
                self.compress_ratio,
                n_pooled,
                overlap=self.compressor.overlap,
            )
            compressor_pooled_positions = _pool_position_ids(
                full_pool_position_ids,
                self.compress_ratio,
                n_pooled,
            )
            if cp_enabled and attn_backend == "tilelang":
                full_kv = _pad_raw_kv_before_compressed_for_tilelang(
                    full_kv,
                    debug_layer_idx=self.layer_idx,
                )
                raw_key_seq_ids = _pad_metadata_to_len(
                    raw_key_seq_ids,
                    full_kv.shape[2],
                    value=-1,
                )
            full_kv = torch.cat([full_kv, pooled], dim=2)
            _dsv4_debug(
                "attn.compressor.forward.after",
                self.layer_idx,
                pooled=pooled,
                indexer_topk=indexer_topk,
                full_kv=full_kv,
                n_pooled=n_pooled,
            )
            _dsv4_debug_backward_hook(pooled, "attn.compressor.pooled", self.layer_idx)
            _dsv4_debug_backward_hook(full_kv, "attn.compressor.full_kv", self.layer_idx)

            # Extend the additive 4D attention mask with a per-query
            # compressed-position mask so dense attention reproduces the
            # reference's ``sparse_attn`` semantics (per-query topk_idxs +
            # causality on the compressed pool).
            #
            # * compress_ratio == 4 (Indexer present): mask=0 only at the
            #   pool positions selected by ``indexer_topk`` for that query,
            #   -inf elsewhere.  ``-1`` entries in ``indexer_topk`` are
            #   already causally-masked by Compressor.
            # * compress_ratio > 4 (no Indexer, e.g. 128): every query q can
            #   attend to compressed position p iff ``p < (q+1) // ratio``
            #   (matches ``get_compress_topk_idxs`` in
            #   ``dsv4flash/inference/model.py:289-296``).
            if attention_mask is not None and n_pooled > 0:
                min_val = torch.finfo(attention_mask.dtype).min
                if indexer_topk is not None:
                    compressed_mask = _build_indexer_topk_compressed_mask(
                        attention_mask,
                        indexer_topk.to(device=full_kv.device),
                        n_pooled,
                    )  # [B, S, P]
                else:
                    p_pos = torch.arange(n_pooled, device=full_kv.device)
                    if query_seq_ids is not None and compressor_pooled_seq_ids is not None:
                        pooled_positions = (
                            compressor_pooled_positions.to(device=full_kv.device, dtype=torch.int64)
                            if compressor_pooled_positions is not None
                            else pooled_seq_positions_from_ids(
                                compressor_pooled_seq_ids.to(device=full_kv.device, dtype=torch.int64)
                            )
                        )
                        query_local_positions = (
                            query_local_position_ids.to(device=full_kv.device, dtype=torch.int64)
                            if query_local_position_ids is not None
                            else query_seq_positions_from_ids(
                                query_seq_ids.to(device=full_kv.device, dtype=torch.int64)
                            )
                        )
                        threshold = ((query_local_positions + 1) // self.compress_ratio).unsqueeze(-1)
                        allowed = (
                            (pooled_positions.unsqueeze(1) >= 0)
                            & (pooled_positions.unsqueeze(1) < threshold)
                            & (
                                compressor_pooled_seq_ids.to(device=full_kv.device, dtype=torch.int64).unsqueeze(1)
                                == query_seq_ids.to(device=full_kv.device, dtype=torch.int64).unsqueeze(-1)
                            )
                            & (query_seq_ids.to(device=full_kv.device, dtype=torch.int64).unsqueeze(-1) >= 0)
                        )
                    else:
                        q_pos = torch.arange(seq_len, device=full_kv.device)
                        threshold = (q_pos + 1) // self.compress_ratio
                        allowed = p_pos.unsqueeze(0) < threshold.unsqueeze(1)  # [S, P]
                    compressed_mask = torch.where(
                        allowed,
                        torch.zeros((), dtype=attention_mask.dtype, device=full_kv.device),
                        torch.full((), min_val, dtype=attention_mask.dtype, device=full_kv.device),
                    )
                    if compressed_mask.dim() == 2:
                        compressed_mask = compressed_mask.expand(batch, seq_len, n_pooled)
                compressed_mask = compressed_mask.unsqueeze(1)  # [B, 1, S, P]
                attention_mask = torch.cat([attention_mask, compressed_mask], dim=-1)

        # If a caller supplied a 4D mask shorter than full_kv but no compressor
        # ran (shouldn't happen, but kept for defense), fall back to neutral pad.
        if attention_mask is not None and full_kv.shape[2] > attention_mask.shape[-1]:
            attention_mask = F.pad(attention_mask, (0, full_kv.shape[2] - attention_mask.shape[-1]), value=0.0)

        if attn_backend == "tilelang":
            # TileLang sparse attention kernels require bf16 Q/KV. Some
            # numerically stable submodules compute in fp32, so normalize the
            # kernel inputs here while keeping the dense torch path unchanged.
            q_sparse = q.to(torch.bfloat16)
            full_kv_sparse = full_kv.to(torch.bfloat16)
            _dsv4_debug_backward_hook(q_sparse, "attn.sparse_attention.q_sparse", self.layer_idx)
            _dsv4_debug_backward_hook(full_kv_sparse, "attn.sparse_attention.full_kv_sparse", self.layer_idx)
            _dsv4_debug(
                "attn.sparse_topk.before",
                self.layer_idx,
                q_sparse=q_sparse,
                full_kv_sparse=full_kv_sparse,
                indexer_topk=indexer_topk,
                n_pooled=n_pooled,
                raw_key_len=full_kv_sparse.shape[2] - n_pooled,
                query_start=query_start,
                query_positions=query_positions,
                raw_key_start=raw_key_start,
            )
            topk_idxs = build_dsv4_sparse_topk_indices(
                batch_size=batch,
                seq_len=seq_len,
                key_len=full_kv_sparse.shape[2],
                window_size=self.sliding_window,
                device=full_kv_sparse.device,
                attention_mask=None if cp_enabled else attention_mask,
                compress_ratio=self.compress_ratio,
                compressed_topk=indexer_topk,
                n_pooled=n_pooled,
                query_start=query_start - raw_key_start,
                query_global_start=query_start,
                query_positions=query_positions,
                query_key_positions=query_key_positions,
                query_local_positions=query_local_position_ids,
                pooled_local_positions=compressor_pooled_positions,
                raw_key_len=full_kv_sparse.shape[2] - n_pooled,
                query_seq_ids=query_seq_ids,
                raw_key_seq_ids=raw_key_seq_ids,
                pooled_seq_ids=compressor_pooled_seq_ids,
            )
            _dsv4_debug("attn.sparse_topk.after", self.layer_idx, topk_idxs=topk_idxs)
            _dsv4_validate_topk_indices(topk_idxs, full_kv_sparse.shape[2], self.layer_idx)
            sinks = self.sinks_param(q_sparse)
            _dsv4_debug(
                "attn.sparse_attention.before",
                self.layer_idx,
                q_kernel=q_sparse,
                kv_kernel=full_kv_sparse,
                sinks=sinks,
                topk_idxs=topk_idxs,
                scaling=self.scaling,
            )
            attn_output = dsv4_sparse_attention(
                # Keep Q as a transpose view for the chunked TileLang path. The
                # wrapper materializes only per-query/head chunks, avoiding a
                # full 2 GiB contiguous Q copy at 128K/CP4.
                q_sparse.transpose(1, 2),
                full_kv_sparse.squeeze(1).contiguous(),
                sinks,
                topk_idxs,
                self.scaling,
                backend=attn_backend,
            )
            _dsv4_debug("attn.sparse_attention.after", self.layer_idx, attn_output=attn_output)
            _dsv4_debug_backward_hook(attn_output, "attn.sparse_attention.output", self.layer_idx)
            if _dsv4_env_flag("DSV4_SYNC_AFTER_SPARSE"):
                torch.cuda.synchronize(attn_output.device)
                _dsv4_debug("attn.sparse_attention.sync.after", self.layer_idx, attn_output=attn_output)
            attn_weights = None
        else:
            _dsv4_debug("attn.eager_attention.before", self.layer_idx, q=q, full_kv=full_kv, attention_mask=attention_mask)
            attn_output, attn_weights = eager_attention_with_sink(
                self,
                q,
                full_kv,
                full_kv,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
            )
            _dsv4_debug("attn.eager_attention.after", self.layer_idx, attn_output=attn_output, attn_weights=attn_weights)
            # eager_attention_with_sink returns [B, S, H, D] (already transposed).

        # Inverse RoPE on the attention output (same (cos, -sin) conjugate pattern
        # HF uses).  Reference: modular_deepseek_v4.py:607.
        attn_output = _apply_partial_rope(attn_output.transpose(1, 2), cos, -sin, self.rope_head_dim).transpose(1, 2)
        _dsv4_debug_backward_hook(attn_output, "attn.post_inverse_rope", self.layer_idx)

        grouped = attn_output.reshape(batch, seq_len, -1).view(batch, seq_len, self.config.o_groups, -1)
        _dsv4_debug("attn.output_projection.before", self.layer_idx, grouped=grouped)
        wo_a_out = self.wo_a(grouped)
        _dsv4_debug("attn.output_projection.after_wo_a", self.layer_idx, wo_a_out=wo_a_out)
        _dsv4_debug_backward_hook(wo_a_out, "attn.output_projection.wo_a_out", self.layer_idx)
        if _dsv4_env_flag("DSV4_SYNC_AFTER_WO_A"):
            torch.cuda.synchronize(wo_a_out.device)
            _dsv4_debug("attn.output_projection.sync_after_wo_a", self.layer_idx, wo_a_out=wo_a_out)
        wo_b_in = wo_a_out.flatten(2)
        _dsv4_debug_backward_hook(wo_b_in, "attn.output_projection.wo_b_in", self.layer_idx)
        try:
            wo_b_seq_chunk = int(os.environ.get("DSV4_OUTPUT_PROJ_SEQ_CHUNK", "0"))
        except ValueError:
            wo_b_seq_chunk = 0
        if wo_b_seq_chunk > 0 and wo_b_in.shape[1] > wo_b_seq_chunk:
            output_chunks = []
            for begin in range(0, wo_b_in.shape[1], wo_b_seq_chunk):
                end = min(begin + wo_b_seq_chunk, wo_b_in.shape[1])
                wo_b_chunk_in = wo_b_in[:, begin:end].contiguous()
                _dsv4_debug(
                    "attn.output_projection.before_wo_b_chunk",
                    self.layer_idx,
                    begin=begin,
                    end=end,
                    wo_b_chunk_in=wo_b_chunk_in,
                )
                output_chunk = self.wo_b(wo_b_chunk_in)
                if _dsv4_env_flag("DSV4_SYNC_AFTER_WO_B_CHUNK"):
                    torch.cuda.synchronize(output_chunk.device)
                    _dsv4_debug(
                        "attn.output_projection.sync_after_wo_b_chunk",
                        self.layer_idx,
                        begin=begin,
                        end=end,
                        output_chunk=output_chunk,
                    )
                _dsv4_debug(
                    "attn.output_projection.after_wo_b_chunk",
                    self.layer_idx,
                    begin=begin,
                    end=end,
                    output_chunk=output_chunk,
                )
                output_chunks.append(output_chunk)
            output = torch.cat(output_chunks, dim=1).contiguous()
        else:
            _dsv4_debug("attn.output_projection.before_wo_b", self.layer_idx, wo_b_in=wo_b_in)
            output = self.wo_b(wo_b_in)
        _dsv4_debug("attn.output_projection.after_wo_b", self.layer_idx, output=output)
        _dsv4_debug_backward_hook(output, "attn.output_projection.output", self.layer_idx)
        if _dsv4_env_flag("DSV4_SYNC_AFTER_OUTPUT_PROJECTION"):
            torch.cuda.synchronize(output.device)
            _dsv4_debug("attn.output_projection.sync.after", self.layer_idx, output=output)
        _dsv4_debug("attn.forward.end", self.layer_idx, output=output)
        return output, attn_weights

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        for linear in (self.wq_a, self.wq_b, self.wkv, self.wo_b, self.wo_a):
            if hasattr(linear, "weight"):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        for norm in (self.q_norm, self.kv_norm):
            norm.reset_parameters()
        nn.init.zeros_(self.sinks_param.weight)
        if self.compressor is not None:
            for mod in self.compressor.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.trunc_normal_(mod.weight, mean=0.0, std=init_std)
            nn.init.zeros_(self.compressor.ape_param.weight)
            if self.compressor.indexer is not None:
                nn.init.zeros_(self.compressor.indexer.ape_param.weight)
