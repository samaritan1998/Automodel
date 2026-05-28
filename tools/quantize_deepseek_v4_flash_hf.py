#!/usr/bin/env python3
"""Export a BF16 DeepSeek-V4 HF checkpoint to the Flash serving layout.

The exporter streams a consolidated HF checkpoint shard-by-shard and uses the
released DeepSeek-V4-Flash checkpoint as the schema source.  Keys that are BF16
in the reference checkpoint are copied as-is.  Dense FP8 weights get e4m3fn
weights plus e8m0 per-128x128 block scales.  Routed expert weights get packed
FP4 e2m1 int8 weights plus e8m0 per-row/per-32-column scales by default, or
FP8 e4m3fn weights plus e8m0 per-128x128 block scales when requested.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FP8_BLOCK = 128
FP8_E4M3_MAX = 448.0
FP4_BLOCK = 32
FP4_E2M1_MAX = 6.0
FP4_THRESHOLDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
REFERENCE_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "I64": torch.int64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--reference-model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-output", action="store_true")
    parser.add_argument("--wait-for-input-seconds", type=int, default=0)
    parser.add_argument("--copy-chat-template", type=Path, default=None)
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--expert-quant", choices=["fp4", "fp8"], default="fp4")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("NODE_RANK", os.environ.get("RANK", "0"))))
    parser.add_argument(
        "--world-size",
        type=int,
        default=int(os.environ.get("NNODES", os.environ.get("WORLD_SIZE", "1"))),
    )
    parser.add_argument("--barrier-timeout-seconds", type=int, default=86400)
    return parser.parse_args()


def load_index(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    with index_path.open() as f:
        return json.load(f)


def wait_for_index(input_dir: Path, timeout: int) -> None:
    deadline = time.time() + timeout
    index_path = input_dir / "model.safetensors.index.json"
    while True:
        if index_path.is_file():
            return
        if timeout <= 0 or time.time() > deadline:
            raise FileNotFoundError(f"Missing input index: {index_path}")
        print(f"Waiting for {index_path} ...", flush=True)
        time.sleep(60)


def validate_safetensors_dir(model_dir: Path) -> None:
    index = load_index(model_dir)
    files = sorted(set(index["weight_map"].values()))
    for filename in files:
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing shard: {path}")
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                f.get_slice(key).get_shape()


def shard_to_keys(weight_map: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        out[filename].append(key)
    return {filename: sorted(keys) for filename, keys in out.items()}


def read_reference_meta(reference_dir: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    index = load_index(reference_dir)
    by_file = shard_to_keys(index["weight_map"])
    meta: dict[str, tuple[str, tuple[int, ...]]] = {}
    for filename, keys in by_file.items():
        with safe_open(reference_dir / filename, framework="pt", device="cpu") as f:
            for key in keys:
                sl = f.get_slice(key)
                meta[key] = (sl.get_dtype(), tuple(sl.get_shape()))
    return meta


def ceil_pow2_scale(max_abs: torch.Tensor, quant_max: float) -> torch.Tensor:
    raw = max_abs.to(torch.float32) / quant_max
    safe = torch.where(raw > 0, raw, torch.ones_like(raw))
    pow2 = torch.pow(2.0, torch.ceil(torch.log2(safe)))
    return torch.where(raw > 0, pow2, torch.ones_like(pow2))


def quantize_fp8_block(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = weight.detach().to(torch.float32).cpu()
    if x.dim() != 2:
        raise ValueError(f"FP8 quantization expects a 2D tensor, got {tuple(x.shape)}")
    rows, cols = x.shape
    brow = math.ceil(rows / FP8_BLOCK)
    bcol = math.ceil(cols / FP8_BLOCK)

    pad_rows = brow * FP8_BLOCK - rows
    pad_cols = bcol * FP8_BLOCK - cols
    if pad_rows or pad_cols:
        x_for_scale = torch.nn.functional.pad(x, (0, pad_cols, 0, pad_rows))
    else:
        x_for_scale = x

    blocks = x_for_scale.view(brow, FP8_BLOCK, bcol, FP8_BLOCK).permute(0, 2, 1, 3)
    scale_f32 = ceil_pow2_scale(blocks.abs().amax(dim=(-1, -2)), FP8_E4M3_MAX)
    scale = scale_f32.to(torch.float8_e8m0fnu)
    scale_decoded = scale.to(torch.float32)
    expanded = scale_decoded.repeat_interleave(FP8_BLOCK, dim=0).repeat_interleave(FP8_BLOCK, dim=1)
    expanded = expanded[:rows, :cols]
    q = (x / expanded).to(torch.float8_e4m3fn)
    return q, scale


def quantize_fp4_expert(weight: torch.Tensor, row_chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = weight.detach().to(torch.float32).cpu()
    if x.dim() != 2:
        raise ValueError(f"FP4 expert quantization expects a 2D tensor, got {tuple(x.shape)}")
    rows, cols = x.shape
    if cols % FP4_BLOCK != 0:
        raise ValueError(f"Expert input dim {cols} is not divisible by {FP4_BLOCK}")
    if cols % 2 != 0:
        raise ValueError(f"Expert input dim {cols} is not divisible by 2 for packing")

    packed = torch.empty((rows, cols // 2), dtype=torch.int8)
    scale = torch.empty((rows, cols // FP4_BLOCK), dtype=torch.float8_e8m0fnu)
    thresholds = FP4_THRESHOLDS

    for start in range(0, rows, row_chunk):
        end = min(start + row_chunk, rows)
        chunk = x[start:end]
        groups = chunk.view(end - start, cols // FP4_BLOCK, FP4_BLOCK)
        scale_f32 = ceil_pow2_scale(groups.abs().amax(dim=-1), FP4_E2M1_MAX)
        scale_q = scale_f32.to(torch.float8_e8m0fnu)
        scale_decoded = scale_q.to(torch.float32).repeat_interleave(FP4_BLOCK, dim=-1)

        normalized = chunk / scale_decoded
        mag_code = torch.bucketize(normalized.abs(), thresholds)
        sign = (normalized < 0) & (mag_code != 0)
        codes = (mag_code + sign.to(torch.long) * 8).to(torch.uint8)
        packed_u8 = codes[:, 0::2] | (codes[:, 1::2] << 4)

        packed[start:end] = packed_u8.contiguous().view(torch.int8)
        scale[start:end] = scale_q

    return packed, scale


def expected_scale_key(key: str) -> str:
    if not key.endswith(".weight"):
        raise ValueError(f"Cannot derive scale key from non-weight key: {key}")
    return key[: -len(".weight")] + ".scale"


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def wait_for_path(path: Path, timeout: int, description: str) -> None:
    deadline = time.time() + timeout
    while not path.exists():
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for {description}: {path}")
        time.sleep(10)


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.rename(path)


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(text)
    tmp.rename(path)


def cast_to_reference_dtype(tensor: torch.Tensor, dtype_name: str) -> torch.Tensor:
    dtype = REFERENCE_DTYPES.get(dtype_name)
    if dtype is None:
        raise RuntimeError(f"Unsupported non-quantized reference dtype {dtype_name}")
    return tensor.to(dtype=dtype).cpu()


def write_metadata(
    input_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    chat_template: Path | None,
    expert_quant: str,
) -> None:
    # The Automodel consolidated checkpoint may contain an internal
    # TokenizersBackend tokenizer, which SGLang can load but which is not the
    # released DeepSeek-V4 tokenizer contract.  Keep serving metadata aligned
    # with the reference Flash checkpoint.
    reference_first_copy_names = [
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ]
    input_first_copy_names = [
        "README.md",
        "LICENSE",
        ".gitattributes",
    ]

    for name in reference_first_copy_names:
        src = reference_dir / name
        if not src.exists():
            src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    for name in input_first_copy_names:
        src = input_dir / name
        if not src.exists():
            src = reference_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    # DeepSeek-V4 serving stacks such as SGLang use the released encoding
    # helper instead of a standard tokenizer chat_template.  Keep it beside
    # the exported weights so deployments can use the official message encoder.
    encoding_dir = reference_dir / "encoding"
    if encoding_dir.is_dir():
        shutil.copytree(encoding_dir, output_dir / "encoding", dirs_exist_ok=True)

    cfg_path = input_dir / "config.json"
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = json.load(f)
    else:
        with (reference_dir / "config.json").open() as f:
            cfg = json.load(f)

    with (reference_dir / "config.json").open() as f:
        ref_cfg = json.load(f)

    # Older consolidated checkpoints may have been written before serving-only
    # metadata such as YaRN rope_scaling was preserved.  Keep train-time
    # structural choices from the input checkpoint, but backfill serving
    # metadata from the released Flash reference when it is absent.
    for metadata_key in ("rope_scaling",):
        if cfg.get(metadata_key) is None and ref_cfg.get(metadata_key) is not None:
            cfg[metadata_key] = ref_cfg[metadata_key]

    if expert_quant == "fp4":
        cfg["expert_dtype"] = "fp4"
    elif expert_quant == "fp8":
        cfg["expert_dtype"] = "fp8"
    else:
        raise ValueError(f"Unsupported expert quantization mode: {expert_quant}")

    cfg["torch_dtype"] = "bfloat16"
    cfg.pop("dtype", None)
    cfg["quantization_config"] = ref_cfg.get(
        "quantization_config",
        {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    )

    with (output_dir / "config.json").open("w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if chat_template is not None:
        if not chat_template.is_file():
            raise FileNotFoundError(f"Missing chat template: {chat_template}")
        shutil.copy2(chat_template, output_dir / "chat_template.jinja")


def check_key_schema(input_keys: Iterable[str], ref_meta: dict[str, tuple[str, tuple[int, ...]]]) -> None:
    input_set = set(input_keys)
    ref_non_scale = {key for key in ref_meta if not key.endswith(".scale")}
    extra = sorted(input_set - ref_non_scale)
    if extra:
        raise RuntimeError(f"Input checkpoint has {len(extra)} unexpected keys, e.g. {extra[:10]}")
    missing = sorted(ref_non_scale - input_set)
    if missing:
        print(
            f"Input checkpoint omits {len(missing)} reference non-scale keys; "
            f"they will not be emitted, e.g. {missing[:10]}",
            flush=True,
        )


def export_quantized(args: argparse.Namespace) -> None:
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError(f"Invalid rank/world_size: rank={args.rank}, world_size={args.world_size}")

    print(
        f"DeepSeek-V4 Flash quant export rank={args.rank}/{args.world_size} host={socket.gethostname()}",
        flush=True,
    )

    wait_for_index(args.input_dir, args.wait_for_input_seconds)

    input_index = load_index(args.input_dir)
    input_weight_map = input_index["weight_map"]
    ref_meta = read_reference_meta(args.reference_model_dir)
    check_key_schema(input_weight_map.keys(), ref_meta)

    tmp_dir = (
        args.output_dir.with_name(args.output_dir.name + ".tmp")
        if args.world_size > 1
        else args.output_dir.with_name(args.output_dir.name + f".tmp-{os.getpid()}")
    )
    parallel_dir = tmp_dir / ".parallel"

    if args.rank == 0:
        validate_safetensors_dir(args.input_dir)
        if args.output_dir.exists():
            if args.overwrite:
                shutil.rmtree(args.output_dir)
            else:
                raise FileExistsError(f"Output dir already exists: {args.output_dir}")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        parallel_dir.mkdir(parents=True)
        write_text_atomic(parallel_dir / "init.done", "ok\n")
    else:
        wait_for_path(parallel_dir / "init.done", args.barrier_timeout_seconds, "rank0 init")

    output_weight_map: dict[str, str] = {}
    total_size = 0
    by_input_shard = sorted(shard_to_keys(input_weight_map).items())
    assigned_shards = [
        (idx, filename, keys)
        for idx, (filename, keys) in enumerate(by_input_shard, start=1)
        if (idx - 1) % args.world_size == args.rank
    ]
    print(
        f"Rank {args.rank}/{args.world_size} assigned {len(assigned_shards)}/{len(by_input_shard)} shards",
        flush=True,
    )

    try:
        for shard_idx, input_filename, keys in assigned_shards:
            output_filename = input_filename
            out_tensors: dict[str, torch.Tensor] = {}
            print(f"[rank {args.rank} shard {shard_idx}/{len(by_input_shard)}] converting {input_filename}", flush=True)

            with safe_open(args.input_dir / input_filename, framework="pt", device="cpu") as f:
                for key in keys:
                    target_dtype, target_shape = ref_meta[key]
                    tensor = f.get_tensor(key)

                    if target_dtype == "F8_E4M3":
                        q, scale = quantize_fp8_block(tensor)
                        scale_key = expected_scale_key(key)
                        scale_dtype, scale_shape = ref_meta[scale_key]
                        if scale_dtype != "F8_E8M0" or tuple(scale.shape) != scale_shape or tuple(q.shape) != target_shape:
                            raise RuntimeError(
                                f"FP8 schema mismatch for {key}: q {tuple(q.shape)} vs {target_shape}, "
                                f"scale {tuple(scale.shape)} vs {scale_shape}/{scale_dtype}"
                            )
                        out_tensors[key] = q
                        out_tensors[scale_key] = scale
                    elif target_dtype == "I8" and args.expert_quant == "fp4":
                        q, scale = quantize_fp4_expert(tensor, args.row_chunk)
                        scale_key = expected_scale_key(key)
                        scale_dtype, scale_shape = ref_meta[scale_key]
                        if scale_dtype != "F8_E8M0" or tuple(scale.shape) != scale_shape or tuple(q.shape) != target_shape:
                            raise RuntimeError(
                                f"FP4 schema mismatch for {key}: q {tuple(q.shape)} vs {target_shape}, "
                                f"scale {tuple(scale.shape)} vs {scale_shape}/{scale_dtype}"
                            )
                        out_tensors[key] = q
                        out_tensors[scale_key] = scale
                    elif target_dtype == "I8" and args.expert_quant == "fp8":
                        q, scale = quantize_fp8_block(tensor)
                        scale_key = expected_scale_key(key)
                        scale_dtype, _ = ref_meta[scale_key]
                        if scale_dtype != "F8_E8M0" or tuple(q.shape) != tuple(tensor.shape):
                            raise RuntimeError(
                                f"FP8 expert schema mismatch for {key}: q {tuple(q.shape)} vs {tuple(tensor.shape)}, "
                                f"scale dtype {scale_dtype}"
                            )
                        out_tensors[key] = q
                        out_tensors[scale_key] = scale
                    else:
                        if tuple(tensor.shape) != target_shape:
                            raise RuntimeError(f"Shape mismatch for {key}: {tuple(tensor.shape)} vs {target_shape}")
                        out_tensors[key] = cast_to_reference_dtype(tensor, target_dtype)

            for out_key, out_tensor in out_tensors.items():
                output_weight_map[out_key] = output_filename
                total_size += tensor_nbytes(out_tensor)
            save_file(out_tensors, tmp_dir / output_filename, metadata={"format": "pt"})

        write_json_atomic(
            parallel_dir / f"rank_{args.rank}.json",
            {"rank": args.rank, "total_size": total_size, "weight_map": output_weight_map},
        )

        if args.rank != 0:
            return

        deadline = time.time() + args.barrier_timeout_seconds
        rank_payloads: list[dict] = []
        pending = set(range(args.world_size))
        while pending:
            for rank in list(pending):
                error_path = parallel_dir / f"rank_{rank}.error"
                if error_path.exists():
                    raise RuntimeError(f"Rank {rank} failed:\n{error_path.read_text()}")
                done_path = parallel_dir / f"rank_{rank}.json"
                if done_path.exists():
                    with done_path.open() as f:
                        rank_payloads.append(json.load(f))
                    pending.remove(rank)
            if pending:
                if time.time() > deadline:
                    raise TimeoutError(f"Timed out waiting for ranks: {sorted(pending)}")
                time.sleep(10)

        merged_weight_map: dict[str, str] = {}
        merged_total_size = 0
        for payload in rank_payloads:
            for key, filename in payload["weight_map"].items():
                if key in merged_weight_map:
                    raise RuntimeError(f"Duplicate output key across ranks: {key}")
                merged_weight_map[key] = filename
            merged_total_size += int(payload["total_size"])

        index = {"metadata": {"total_size": merged_total_size}, "weight_map": dict(sorted(merged_weight_map.items()))}
        write_json_atomic(tmp_dir / "model.safetensors.index.json", index)

        write_metadata(args.input_dir, args.reference_model_dir, tmp_dir, args.copy_chat_template, args.expert_quant)

        if args.validate_output:
            validate_safetensors_dir(tmp_dir)

        shutil.rmtree(parallel_dir)
        tmp_dir.rename(args.output_dir)
    except Exception as exc:
        if parallel_dir.exists():
            write_text_atomic(parallel_dir / f"rank_{args.rank}.error", f"{type(exc).__name__}: {exc}\n")
        print(f"Quantized export failed; partial output kept at {tmp_dir}", flush=True)
        raise


def main() -> None:
    args = parse_args()
    export_quantized(args)


if __name__ == "__main__":
    main()
