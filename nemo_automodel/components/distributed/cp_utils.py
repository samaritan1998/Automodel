# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib
import os
from typing import List, Optional, Set

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.thd_utils import split_batch_into_thd_chunks


def _build_position_ids(batch, device):
    """Add position_ids to the batch only if they are missing."""
    # TODO(@boxiangw): Refractor. Needed for SP support
    # If 'position_ids' does not exist in batch already then override it.
    # In case of Packed sequence contains 'position_ids' and we don't want to override it.
    if "position_ids" not in batch:
        seq_len = batch["input_ids"].shape[1]
        batch["position_ids"] = torch.arange(seq_len, device=device).unsqueeze(0)
    return batch


# based on https://github.com/pytorch/torchtitan/blob/0b44d4c437c424b6bf719661c0eb4283dc4068bc/torchtitan/distributed/utils.py#L180  # pylint: disable=C0301
def get_train_context(enable_loss_parallel: bool, enable_compiled_autograd: bool, cp_context=None):
    """
    Create a train context.

    Args:
        enable_loss_parallel (bool): Whether to enable loss parallelism.
        enable_compiled_autograd (bool): Whether to enable compiled autograd.
    """

    @contextlib.contextmanager
    def context():
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            if enable_compiled_autograd:
                stack.enter_context(torch._dynamo.utils.maybe_enable_compiled_autograd(True))

            if cp_context is not None:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                # currently we only support these two SDP backends.
                # SDPBackend.MATH is not currently compatible with DTensor
                stack.enter_context(sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]))
                stack.enter_context(cp_context)

            yield

    return context


# based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py#L113
def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: List[torch.Tensor],
    cp_seq_dims: List[int],
    cp_no_restore_buffers: Set[torch.Tensor],
    cp_rotate_method: Optional[str] = None,
):
    """
    Create a context parallel context.

    Args:
        cp_mesh (DeviceMesh): The device mesh for context parallel.
        cp_buffers (List[torch.Tensor]): The buffers for context parallel.
        cp_seq_dims (List[int]): The sequence dimensions for context parallel.
        cp_no_restore_buffers (Set[torch.Tensor]): The no restore buffers for context parallel.
        cp_rotate_method (str): The rotation method for context parallel,
            such as "allgather" or "addtoall".
    """
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor.experimental._attention import set_rotate_method

    if cp_rotate_method is not None:
        set_rotate_method(cp_rotate_method)

    # TODO: uncomment this when torch.distributed.tensor.experimental._attention.set_rotate_method
    # is available
    # from torch.distributed.tensor.experimental._attention import set_rotate_method
    # set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


def attach_context_parallel_hooks(model: torch.nn.Module):
    """Attach forward pre-hooks to self_attn modules to fix attention masks for context parallelism.

    Context parallelism shards Q/K/V on the sequence dimension as DTensors,
    so explicit 4D attention masks would have mismatched shapes.  This function
    registers a hook on every ``self_attn`` sub-module that strips the
    ``attention_mask`` kwarg and sets ``is_causal=True`` instead, letting
    SDPA handle causal masking internally.

    Based on ``accelerate.big_modeling._attach_context_parallel_hooks``.
    """

    def _self_attn_pre_forward_hook(_module, module_args, module_kwargs):
        if "attention_mask" in module_kwargs:
            module_kwargs["attention_mask"] = None
            module_kwargs["is_causal"] = True
        return module_args, module_kwargs

    for name, module in model.named_modules():
        if name.endswith("self_attn"):
            module.register_forward_pre_hook(_self_attn_pre_forward_hook, with_kwargs=True, prepend=True)


def attach_cp_sdpa_hooks(model: torch.nn.Module, cp_mesh) -> None:
    """Inject CP-aware SDPA into self_attn modules for compile + CP>1 correctness.

    Problem: when per-layer torch.compile is active, Dynamo traces through the decoder
    layer including Q/K/V projections.  At the F.scaled_dot_product_attention call site,
    Q/K/V are already local tensors (DTensor metadata was never propagated through the
    compiled graph).  The DTensor SDPA dispatch — which triggers the CP allgather — never
    fires, so each rank silently attends only to its local sequence shard.

    Fix: swap F.scaled_dot_product_attention with a @torch._dynamo.disable wrapper for
    the duration of each self_attn forward.  Dynamo sees the disabled function and creates
    a graph break there, so:
      - Everything before (Q/K/V proj + RoPE) is compiled and fused.
      - The disabled wrapper runs eagerly: re-wraps local Q/K/V as DTensors with
        Shard(2) on the CP mesh so the DTensor SDPA dispatch fires the allgather.
      - Everything after (O proj + residual + MLP) is compiled and fused.

    Seq dim at the SDPA call is 2: tensors are [B, nH, S/cp_size, D] after HF reshape.
    """
    import torch.nn.functional as F_module
    from torch.distributed.tensor import DTensor, Shard

    _original_sdpa = F_module.scaled_dot_product_attention

    @torch._dynamo.disable
    def _cp_sdpa(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kwargs
    ):
        # Re-wrap local Q/K/V as DTensors so DTensor SDPA dispatch fires the CP allgather.
        # Seq dim is 2: [B, nH, S/cp_size, D].
        if not isinstance(query, DTensor):
            query = DTensor.from_local(query, device_mesh=cp_mesh, placements=[Shard(2)])
            key = DTensor.from_local(key, device_mesh=cp_mesh, placements=[Shard(2)])
            value = DTensor.from_local(value, device_mesh=cp_mesh, placements=[Shard(2)])
        out = _original_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
            **kwargs,
        )
        # Unwrap back to local tensor for the compiled O-proj + MLP region.
        return out.to_local() if isinstance(out, DTensor) else out

    def _pre_hook(module, args, kwargs):
        F_module.scaled_dot_product_attention = _cp_sdpa
        return args, kwargs

    def _post_hook(module, inputs, output):
        F_module.scaled_dot_product_attention = _original_sdpa

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    for name, module in model.named_modules():
        if name.endswith("self_attn"):
            # Hook on the inner attention module so the hook fires during both
            # the original forward AND gradient-checkpointing recompute.
            # CheckpointWrapper's recompute bypasses __call__ (and thus pre-hooks
            # on the wrapper itself), so we must hook on the wrapped module directly.
            target = module._checkpoint_wrapped_module if isinstance(module, CheckpointWrapper) else module
            target.register_forward_pre_hook(_pre_hook, with_kwargs=True)
            # always_call=True ensures _original_sdpa is restored even if the forward raises.
            target.register_forward_hook(_post_hook, always_call=True)


def make_cp_batch_and_ctx(
    device_mesh,
    batch,
    loss_mask=None,
    use_te: bool = False,
    use_dsv4_cp: bool = False,
    padding_token_id: int = 0,
    num_chunks: int = 1,
    seq_lens_padding_value: int = -1000,
):
    """
    Build a CP context manager and shards a batch. If the input device_mesh is None or the size
    of the context_parallel submesh is 1, this function is effectively a no-op.

    Args:
        cp_mesh (DeviceMesh): The device mesh for context parallel.
        batch (Dict[str, torch.Tensor]): The input batch containing (string, torch.Tensor)

    Returns:
        tuple (contextmanager, dict[str, torch.Tensor]): Returns a tuple with a context manager
        and a new batch. The context manager is either nullcontext (no CP) or CP context manager as
        returned by `create_context_parallel_ctx`. The batch has also been passed to
        `create_context_parallel_ctx` and is accordingly sharded.
    """
    from contextlib import nullcontext

    def _get_submesh(device_mesh, name):
        if name in getattr(device_mesh, "mesh_dim_names", {}):
            return device_mesh[name]
        return None

    def _get_mesh_size(mesh):
        if mesh is None:
            return 0
        return mesh.size()

    cp_mesh = _get_submesh(device_mesh, "cp")
    tp_mesh = _get_submesh(device_mesh, "tp")

    if use_te:
        return nullcontext, make_cp_batch_for_te(
            cp_mesh,
            batch,
            padding_token_id=padding_token_id,
            qkv_format="thd",
            num_chunks=num_chunks,
            seq_lens_padding_value=seq_lens_padding_value,
        )

    if _get_mesh_size(cp_mesh) <= 1:
        return nullcontext, batch

    if use_dsv4_cp:
        return nullcontext, make_cp_batch_for_dsv4(
            cp_mesh,
            batch,
            padding_token_id=padding_token_id,
        )

    # Remove attention_mask from the batch so the model does not attempt to
    # build a 4D causal mask (which would have mismatched shapes with
    # DTensor-sharded Q/K/V).  Each self_attn module's forward_pre_hook
    # (registered by attach_context_parallel_hooks) will set is_causal=True
    # so that SDPA handles causal masking internally.
    batch.pop("attention_mask", None)

    # Determine the primary sequence tensor: inputs_embeds (VLM with CP, where
    # multimodal token replacement happened pre-shard) or input_ids (standard LLM).
    has_inputs_embeds = "inputs_embeds" in batch
    has_input_ids = "input_ids" in batch
    assert has_inputs_embeds ^ has_input_ids, (
        "make_cp_batch_and_ctx requires exactly one of 'inputs_embeds' or 'input_ids' in batch"
    )
    if has_inputs_embeds:
        primary_seq_tensor = batch["inputs_embeds"]
    else:
        primary_seq_tensor = batch["input_ids"]
    seq_len = primary_seq_tensor.shape[1]

    # Skip 1D injection if position_ids already in batch (e.g. mRoPE pre-computed)
    batch_size = primary_seq_tensor.shape[0]
    if "position_ids" not in batch and (_get_mesh_size(cp_mesh) > 1 or _get_mesh_size(tp_mesh) > 1):
        batch["position_ids"] = (
            torch.arange(0, seq_len, device=primary_seq_tensor.device).unsqueeze(0).expand(batch_size, -1).contiguous()
        )
    elif "position_ids" in batch:
        position_ids = batch["position_ids"]
        if position_ids.ndim == 2 and position_ids.shape[0] == 1 and batch_size > 1:
            batch["position_ids"] = position_ids.expand(batch_size, -1).contiguous()

    position_ids = batch["position_ids"]

    # Determine correct seq dim for CP sharding
    # mRoPE: [3, B, S] → shard on dim 2; standard: [B, S] → shard on dim 1
    pos_seq_dim = 2 if position_ids.ndim == 3 else 1

    labels = batch["labels"]

    # Collect all available tensors for context parallel.  We track each
    # cp_buffer's batch key (when sourced from ``batch``) so the padding pass
    # below can pick the semantically-correct fill sentinel and mirror the
    # padded tensor back into ``batch``.  ``loss_mask`` is passed as an arg
    # (not in batch) so it has no key.
    primary_key = "inputs_embeds" if has_inputs_embeds else "input_ids"
    cp_buffers = [primary_seq_tensor, labels, position_ids]
    # inputs_embeds is [B, S, H] → seq_dim=1; input_ids is [B, S] → seq_dim=1
    cp_seq_dims = [1, 1, pos_seq_dim]
    cp_no_restore_buffers = {primary_seq_tensor, labels}
    batch_buffer_keys: dict[int, str] = {0: primary_key, 1: "labels", 2: "position_ids"}

    # Add loss_mask if available (passed as arg, not in batch -> no key)
    if loss_mask is not None:
        cp_buffers.append(loss_mask)
        cp_seq_dims.append(1)
        cp_no_restore_buffers.add(loss_mask)

    # Add padding_mask if available in batch
    if "padding_mask" in batch:
        padding_mask = batch["padding_mask"]
        batch_buffer_keys[len(cp_buffers)] = "padding_mask"
        cp_buffers.append(padding_mask)
        cp_seq_dims.append(1)
        cp_no_restore_buffers.add(padding_mask)

    # Pad sequence length to be divisible by 2 * cp_size (required by
    # context_parallel load balancing). The inputs_embeds path can hit
    # arbitrary seq lengths from the VLM collator, so we pad here rather
    # than relying on dataset-side padding.
    #
    # Per-buffer pad sentinels: each tensor's "ignore" value is semantic, not
    # dtype-derived.  ``labels``/``padding_mask``/``attention_mask`` are all
    # int/bool but have different ignore conventions.  Falling through to 0
    # for ``padding_mask`` (== False == "real token") would tell the MoE
    # router to route the cp-pad slots to experts -- silently wasting capacity
    # and skewing load-balance loss.
    PAD_FILL = {
        "labels": -100,  # CE ignore_index
        "padding_mask": True,  # bool: True == "this position is pad, ignore"
        "attention_mask": False,  # HF: 0 == "this position is pad, ignore"
        # everything else (input_ids, position_ids, ...) -> 0
    }
    cp_divisor = cp_mesh.size() * 2
    if seq_len % cp_divisor != 0:
        pad_len = cp_divisor - (seq_len % cp_divisor)
        new_no_restore = set()
        for i, (buf, dim) in enumerate(zip(cp_buffers, cp_seq_dims)):
            pad_shape = list(buf.shape)
            pad_shape[dim] = pad_len
            if buf.dtype.is_floating_point:
                pad_val = torch.zeros(pad_shape, dtype=buf.dtype, device=buf.device)
            else:
                fill_val = PAD_FILL.get(batch_buffer_keys.get(i), 0)
                pad_val = torch.full(pad_shape, fill_val, dtype=buf.dtype, device=buf.device)
            old_buf = buf
            cp_buffers[i] = torch.cat([buf, pad_val], dim=dim)
            if old_buf in cp_no_restore_buffers:
                new_no_restore.add(cp_buffers[i])
        cp_no_restore_buffers = new_no_restore
        # Mirror every batch-sourced cp_buffer back into ``batch`` so any
        # downstream consumer reading from the dict sees the padded shape.
        for idx, key in batch_buffer_keys.items():
            batch[key] = cp_buffers[idx]

    cp_ctx = create_context_parallel_ctx(
        cp_mesh=cp_mesh,
        cp_buffers=cp_buffers,
        cp_seq_dims=cp_seq_dims,
        cp_no_restore_buffers=cp_no_restore_buffers,
        cp_rotate_method="allgather",  # TODO: expose through cfg
    )
    # TODO(@akoumparouli): surface these in the future.
    enable_loss_parallel: bool = False
    enable_compiled_autograd: bool = False
    return get_train_context(enable_loss_parallel, enable_compiled_autograd, cp_ctx), batch


def _pad_tensor_along_dim(tensor: torch.Tensor, pad_len: int, dim: int, value) -> torch.Tensor:
    if pad_len <= 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_len
    pad = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=dim)


def _dsv4_packed_alignment() -> int:
    value = os.environ.get("DSV4_PACKED_ALIGNMENT", "128")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"DSV4_PACKED_ALIGNMENT must be an integer, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"DSV4_PACKED_ALIGNMENT must be >= 1, got {parsed}")
    return parsed


def _validate_dsv4_packed_metadata(
    *,
    seq_ids: torch.Tensor,
    labels: torch.Tensor,
    position_ids: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> None:
    padding_from_seq_ids = seq_ids < 0
    if padding_mask is not None and not torch.equal(padding_mask.to(torch.bool), padding_from_seq_ids):
        raise ValueError("DeepSeek V4 packed manual CP requires padding_mask to equal (seq_ids < 0)")
    if padding_from_seq_ids.any() and not torch.all(labels[padding_from_seq_ids] == -100):
        raise ValueError("DeepSeek V4 packed manual CP requires labels to be -100 where seq_ids < 0")

    alignment = _dsv4_packed_alignment()
    absolute = torch.arange(seq_ids.shape[1], device=seq_ids.device, dtype=torch.int64)
    for batch_idx in range(seq_ids.shape[0]):
        row = seq_ids[batch_idx].to(torch.int64)
        valid = row >= 0
        if not valid.any():
            continue

        prev = torch.cat([row.new_full((1,), -1), row[:-1]])
        starts = valid & (row != prev)
        start_positions = torch.nonzero(starts, as_tuple=False).flatten()
        if alignment > 1 and start_positions.numel() and torch.any(start_positions % alignment != 0):
            bad_start = int(start_positions[start_positions % alignment != 0][0].item())
            raise ValueError(
                "DeepSeek V4 packed manual CP requires each sample start to align to "
                f"{alignment} tokens; first bad start={bad_start}. "
                "Use THD packing with seq_pad_multiple >= 128."
            )

        start_ids = row[start_positions]
        if torch.unique(start_ids).numel() != start_ids.numel():
            raise ValueError("DeepSeek V4 packed manual CP requires each seq_id to form one contiguous block")

        start_markers = torch.where(starts, absolute, torch.zeros_like(absolute))
        start_offsets = torch.cummax(start_markers, dim=0).values
        expected_positions = absolute - start_offsets
        actual_positions = position_ids[batch_idx].to(torch.int64)
        if not torch.equal(actual_positions[valid], expected_positions[valid]):
            raise ValueError(
                "DeepSeek V4 packed manual CP requires position_ids to start at 0 and increment within each seq_id"
            )


def _validate_dsv4_manual_cp_batch(batch, seq_len: int, *, packed_thd: bool = False) -> None:
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    position_ids = batch["position_ids"]
    if input_ids.ndim != 2 or labels.ndim != 2 or position_ids.ndim != 2:
        raise ValueError("DeepSeek V4 manual CP supports only 2D BSH-style input_ids, labels, and position_ids")
    if labels.shape != input_ids.shape or position_ids.shape != input_ids.shape:
        raise ValueError("DeepSeek V4 manual CP requires input_ids, labels, and position_ids to have the same shape")
    if packed_thd:
        if "seq_ids" not in batch:
            raise ValueError("DeepSeek V4 packed manual CP requires seq_ids from packed_sequence_thd_collater")
        seq_ids = batch["seq_ids"]
        if not isinstance(seq_ids, torch.Tensor) or seq_ids.ndim != 2:
            raise ValueError("DeepSeek V4 packed manual CP supports only 2D seq_ids")
        if seq_ids.shape != input_ids.shape:
            raise ValueError("DeepSeek V4 packed manual CP requires seq_ids to match input_ids shape")

    valid_mask = None
    padding_mask = None
    if "attention_mask" in batch:
        attention_mask = batch["attention_mask"]
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
            raise ValueError("DeepSeek V4 manual CP supports only 2D right-padded attention_mask")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("DeepSeek V4 manual CP requires attention_mask to match input_ids shape")
        valid_mask = attention_mask.to(torch.bool)
        padding_mask = valid_mask.logical_not()
    elif "padding_mask" in batch:
        padding_mask = batch["padding_mask"]
        if not isinstance(padding_mask, torch.Tensor) or padding_mask.ndim != 2:
            raise ValueError("DeepSeek V4 manual CP supports only 2D right-padded padding_mask")
        if padding_mask.shape != input_ids.shape:
            raise ValueError("DeepSeek V4 manual CP requires padding_mask to match input_ids shape")
        padding_mask = padding_mask.to(torch.bool)
        valid_mask = padding_mask.logical_not()

    if packed_thd:
        _validate_dsv4_packed_metadata(
            seq_ids=seq_ids,
            labels=labels,
            position_ids=position_ids,
            padding_mask=padding_mask,
        )
        return

    expected = torch.arange(seq_len, device=position_ids.device).unsqueeze(0).expand_as(position_ids)
    if valid_mask is not None:
        valid_int = valid_mask.to(torch.int8)
        if (valid_int[:, 1:] > valid_int[:, :-1]).any():
            raise ValueError("DeepSeek V4 manual CP supports only right padding; left/internal padding is unsupported")
        if not torch.equal(position_ids[valid_mask], expected[valid_mask]):
            raise ValueError("DeepSeek V4 manual CP requires contiguous position_ids over valid tokens")
    elif not torch.equal(position_ids, expected):
        raise ValueError("DeepSeek V4 manual CP requires contiguous position_ids when no padding mask is present")


def _dsv4_cp_layout() -> str:
    return os.environ.get("DSV4_CP_LAYOUT", "contiguous").strip().lower().replace("-", "_")


def _dsv4_zigzag_indices(seq_len: int, cp_size: int, cp_rank: int, device: torch.device) -> torch.Tensor:
    if seq_len % (2 * cp_size) != 0:
        raise ValueError(
            "DeepSeek V4 zigzag CP requires sequence length to be divisible by 2*cp_size "
            f"(seq_len={seq_len}, cp_size={cp_size})"
        )
    half_len = seq_len // (2 * cp_size)
    first_start = cp_rank * half_len
    second_start = (2 * cp_size - cp_rank - 1) * half_len
    first = torch.arange(first_start, first_start + half_len, device=device)
    second = torch.arange(second_start, second_start + half_len, device=device)
    return torch.cat((first, second), dim=0)


def make_cp_batch_for_dsv4(cp_mesh, batch, padding_token_id: int = 0):
    """Manually sequence-shard a DeepSeek V4 batch for sparse-attention CP."""
    if cp_mesh is None or cp_mesh.size() <= 1:
        return batch
    if "input_ids" not in batch or "labels" not in batch:
        raise ValueError("DeepSeek V4 manual CP requires input_ids and labels")

    packed_thd = batch.get("qkv_format") == "thd"
    cp_size = cp_mesh.size()
    cp_rank = torch.distributed.get_rank(group=cp_mesh.get_group())
    input_ids = batch["input_ids"]
    seq_len = input_ids.shape[1]
    batch["dsv4_token_positions"] = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(
        input_ids.shape[0],
        -1,
    )
    if "attention_mask" in batch and "padding_mask" not in batch:
        batch["padding_mask"] = batch["attention_mask"].to(torch.bool).logical_not()

    cp_layout = _dsv4_cp_layout()
    if cp_layout not in {"contiguous", "zigzag"}:
        raise ValueError(f"Unsupported DeepSeek V4 CP layout: {cp_layout!r}")
    divisibility = 2 * cp_size if cp_layout == "zigzag" else cp_size
    pad_len = (-seq_len) % divisibility

    if "position_ids" not in batch:
        batch["position_ids"] = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(
            input_ids.shape[0],
            -1,
        )

    _validate_dsv4_manual_cp_batch(batch, seq_len, packed_thd=packed_thd)

    if pad_len:
        batch["input_ids"] = _pad_tensor_along_dim(batch["input_ids"], pad_len, 1, padding_token_id)
        batch["labels"] = _pad_tensor_along_dim(batch["labels"], pad_len, 1, -100)
        batch["position_ids"] = _pad_tensor_along_dim(batch["position_ids"], pad_len, 1, 0)
        batch["dsv4_token_positions"] = _pad_tensor_along_dim(batch["dsv4_token_positions"], pad_len, 1, 0)
        if "attention_mask" in batch:
            batch["attention_mask"] = _pad_tensor_along_dim(batch["attention_mask"], pad_len, 1, 0)
        if "padding_mask" in batch:
            batch["padding_mask"] = _pad_tensor_along_dim(batch["padding_mask"], pad_len, 1, True)
        if "seq_ids" in batch:
            batch["seq_ids"] = _pad_tensor_along_dim(batch["seq_ids"], pad_len, 1, -1)

    padded_seq_len = batch["input_ids"].shape[1]
    cp_sharded_keys = (
        "input_ids",
        "labels",
        "position_ids",
        "attention_mask",
        "padding_mask",
        "seq_ids",
        "dsv4_token_positions",
    )
    if cp_layout == "zigzag":
        indices = _dsv4_zigzag_indices(padded_seq_len, cp_size, cp_rank, input_ids.device)
        for key in cp_sharded_keys:
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].index_select(1, indices).contiguous()
    else:
        shard_len = padded_seq_len // cp_size
        start = cp_rank * shard_len
        end = start + shard_len
        for key in cp_sharded_keys:
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key][:, start:end].contiguous()

    # DSV4 sparse attention builds its own causal/top-k indices from seq ids
    # and position ids. Keep padding_mask for MoE; remove dense attention_mask.
    batch.pop("attention_mask", None)
    if "seq_ids" in batch:
        batch["dsv4_seq_ids"] = batch["seq_ids"]
    batch["dsv4_cp_size"] = cp_size
    batch["dsv4_cp_rank"] = cp_rank
    batch["dsv4_cp_layout"] = cp_layout
    return batch


def make_cp_batch_for_te(
    cp_mesh,
    batch,
    qkv_format="thd",
    padding_token_id: int = 0,
    num_chunks: int = 1,
    seq_lens_padding_value: int = -1000,
):
    """
    Build a CP batch for Transformer Engine using THD format.

    This function converts BSHD format batches to THD format and shards them across
    context parallel ranks for use with Transformer Engine. It processes the batch
    in chunks if num_chunks > 1, allowing for better memory efficiency with large
    sequences.

    The function performs three main steps:
    1. Converts BSHD format to THD format using split_batch_into_thd_chunks
    2. Optionally splits the batch into multiple chunks for memory efficiency
    3. Shards each chunk across CP ranks using Transformer Engine's partitioning

    Args:
        cp_mesh (DeviceMesh or None): The device mesh for context parallel. If None or
            size <= 1, returns the batch in THD format without sharding.
        batch (Dict[str, torch.Tensor]): The input batch in BSHD format containing:
            - input_ids: Input token IDs [batch_size, seq_len] or [batch_size, seq_len, hidden_dim]
            - labels: Label token IDs [batch_size, seq_len]
            - position_ids (optional): Position IDs [batch_size, seq_len]
            - seq_lens: Actual sequence lengths [batch_size, num_packs]
            - seq_lens_padded: Padded sequence lengths [batch_size, num_packs]
        qkv_format (str): Format for QKV tensors. Currently only "thd" is supported.
        padding_token_id (int): Token ID used for padding in input_ids (default: 0)
        num_chunks (int): Number of chunks to split the batch into. If > 1, the batch
            dimension is split and each chunk is processed separately (default: 1)
        seq_lens_padding_value (int): Sentinel value used to indicate padding in
            seq_lens/seq_lens_padded tensors (default: -1000)

    Returns:
        dict: Processed batch in THD format with the following keys:
            - input_ids: Sharded input token IDs [total_tokens] or [num_chunks, chunk_tokens]
            - labels: Sharded labels [total_tokens] or [num_chunks, chunk_tokens]
            - position_ids: Generated and sharded position IDs [total_tokens] or [num_chunks, chunk_tokens]
            - cu_seqlens: Cumulative sequence lengths [num_seqs+1] or [num_chunks, max_seqs+1]
            - cu_seqlens_padded: Cumulative padded sequence lengths [num_seqs+1] or [num_chunks, max_seqs+1]
            - max_seqlen: Maximum sequence length (int32 tensor)
            - qkv_format: Format string ("thd")
            - padding_mask: Boolean mask indicating padding tokens

    Raises:
        ValueError: If qkv_format is not "thd"
        KeyError: If required fields (seq_lens, seq_lens_padded) are missing from batch

    Example:
        >>> # Single chunk, no CP
        >>> batch = {
        ...     'input_ids': torch.tensor([[1, 2, 3, 4]]),
        ...     'labels': torch.tensor([[2, 3, 4, 5]]),
        ...     'seq_lens': torch.tensor([[4]]),
        ...     'seq_lens_padded': torch.tensor([[4]])
        ... }
        >>> result = make_cp_batch_for_te(None, batch)
        >>> result['input_ids'].shape  # [4] in THD format
        torch.Size([4])

        >>> # Multiple chunks with CP
        >>> batch = {
        ...     'input_ids': torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        ...     'labels': torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]]),
        ...     'seq_lens': torch.tensor([[4], [4]]),
        ...     'seq_lens_padded': torch.tensor([[4], [4]])
        ... }
        >>> result = make_cp_batch_for_te(cp_mesh, batch, num_chunks=2)
        >>> result['input_ids'].shape  # [2, chunk_tokens] - 2 chunks
        torch.Size([2, 2])  # Example: 2 chunks, 2 tokens each after sharding
    """
    if qkv_format != "thd":
        raise ValueError(f"Currently only 'thd' format is supported, got: {qkv_format}")

    batch = split_batch_into_thd_chunks(
        batch, num_chunks=num_chunks, seq_lens_padding_value=seq_lens_padding_value, padding_token_id=padding_token_id
    )

    if cp_mesh is None or cp_mesh.size() <= 1:
        return batch

    if num_chunks <= 1:
        return _shard_thd_chunk_for_te(batch, cp_mesh, qkv_format, seq_lens_padding_value, padding_token_id)

    # Extract each chunk from the batched result and shard it
    chunks = []
    for i in range(num_chunks):
        chunk_batch = {k: v[i] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        chunks.append(
            _shard_thd_chunk_for_te(chunk_batch, cp_mesh, qkv_format, seq_lens_padding_value, padding_token_id)
        )

    return_dict = {
        "input_ids": torch.stack([chunk["input_ids"] for chunk in chunks]),
        "labels": torch.stack([chunk["labels"] for chunk in chunks]),
        "position_ids": torch.stack([chunk["position_ids"] for chunk in chunks]),
        "cu_seqlens": torch.stack([chunk["cu_seqlens"] for chunk in chunks]),
        "max_seqlen": torch.stack([chunk["max_seqlen"] for chunk in chunks]),
        "qkv_format": qkv_format,
        "padding_mask": torch.stack([chunk["padding_mask"] for chunk in chunks]),
        "cp_size": cp_mesh.size() if cp_mesh is not None else 1,
        "cp_rank": torch.distributed.get_rank(group=cp_mesh.get_group()) if cp_mesh is not None else 0,
    }

    return return_dict


def _shard_thd_chunk_for_te(
    batch,
    cp_mesh,
    qkv_format,
    seq_lens_padding_value,
    padding_token_id,
):
    import transformer_engine_torch as tex

    cu_seqlens = batch.get("cu_seqlens", None)
    cu_seqlens_padded = batch.get("cu_seqlens_padded", batch["cu_seqlens"])
    filtered_cu_seqlens_padded = cu_seqlens_padded[cu_seqlens_padded != seq_lens_padding_value]

    # Check for required fields - BSHD format is not supported
    if cu_seqlens is None or cu_seqlens_padded is None:
        raise ValueError(
            "BSHD format is not supported. Both 'cu_seqlens' and 'cu_seqlens_padded' must be present in the batch. "
            "Please use packed sequence format with cu_seqlens and cu_seqlens_padded."
        )

    cp_size = cp_mesh.size()

    cp_rank = torch.distributed.get_rank(group=cp_mesh.get_group()) if cp_mesh is not None else 0

    # Handle all mask keys that may be present in the batch
    mask_keys = ["input_ids", "labels", "position_ids", "padding_mask"]

    for key in mask_keys:
        if key in batch:
            val = batch[key]
            index = tex.thd_get_partitioned_indices(filtered_cu_seqlens_padded, val.size(0), cp_size, cp_rank)
            val = val.index_select(0, index)
            batch[key] = val

    max_seqlen = (filtered_cu_seqlens_padded[1:] - filtered_cu_seqlens_padded[:-1]).max().item()
    if "padding_mask" in batch and batch["padding_mask"] is not None:
        padding_mask = batch["padding_mask"].to(torch.bool).contiguous()
    elif "attention_mask" in batch and batch["attention_mask"] is not None:
        padding_mask = batch["attention_mask"].to(torch.bool).logical_not().contiguous()
    elif batch["input_ids"].ndim == 1:
        padding_mask = (batch["input_ids"] == padding_token_id).bool().contiguous()
    else:
        padding_mask = torch.zeros(batch["input_ids"].shape[0], dtype=torch.bool, device=batch["input_ids"].device)
    output_batch = {
        "input_ids": batch["input_ids"].to(torch.int64).contiguous(),
        "labels": batch["labels"].to(torch.int64).contiguous(),
        "position_ids": batch["position_ids"].to(torch.int64).contiguous(),
        "cu_seqlens": cu_seqlens_padded.to(torch.int32).contiguous(),
        "max_seqlen": torch.tensor(max_seqlen).to(torch.int32).to(device=cu_seqlens_padded.device),
        "qkv_format": qkv_format,
        "padding_mask": padding_mask,
        "cp_size": cp_size,
        "cp_rank": cp_rank,
    }

    return output_batch
