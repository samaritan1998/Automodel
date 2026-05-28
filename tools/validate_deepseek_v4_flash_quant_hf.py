#!/usr/bin/env python3
"""Validate a DeepSeek-V4 BF16 HF checkpoint against a quantized HF export.

The script is intentionally file based so it can run as an LTP multi-node job
without initializing torch.distributed.  Each rank checks a subset of input
safetensor shards, dequantizes the corresponding quantized tensors, and writes
per-tensor error statistics.  Rank 0 merges the JSONL reports into a compact
summary.
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
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


FP8_BLOCK = 128
FP4_BLOCK = 32
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="BF16 consolidated HF checkpoint")
    parser.add_argument("--quant-dir", required=True, type=Path, help="Quantized HF checkpoint to validate")
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("NODE_RANK", os.environ.get("RANK", "0"))))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("NNODES", os.environ.get("WORLD_SIZE", "1"))))
    parser.add_argument("--barrier-timeout-seconds", type=int, default=86400)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--warn-rel-mean", type=float, default=0.02)
    parser.add_argument("--warn-max-abs", type=float, default=0.25)
    parser.add_argument("--max-shards", type=int, default=0, help="Debug limit per rank; 0 means all assigned shards")
    parser.add_argument("--start-shard-index", type=int, default=1, help="1-based global shard index lower bound")
    return parser.parse_args()


def load_index(model_dir: Path) -> dict[str, Any]:
    with (model_dir / "model.safetensors.index.json").open() as f:
        return json.load(f)


def shard_to_keys(weight_map: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        out[filename].append(key)
    return {filename: sorted(keys) for filename, keys in out.items()}


def write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.rename(path)


def wait_for_path(path: Path, timeout: int, description: str) -> None:
    deadline = time.time() + timeout
    while not path.exists():
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for {description}: {path}")
        time.sleep(10)


def scale_to_f32(scale: torch.Tensor) -> torch.Tensor:
    return scale.to(torch.float32)


def dequant_fp8_rows(weight: torch.Tensor, scale: torch.Tensor, start_row: int, rows: int, cols: int) -> torch.Tensor:
    scale_f32 = scale_to_f32(scale)
    expanded = scale_f32.repeat_interleave(FP8_BLOCK, dim=0).repeat_interleave(FP8_BLOCK, dim=1)
    local_start = start_row % FP8_BLOCK
    expanded = expanded[local_start : local_start + rows, :cols]
    return weight.to(torch.float32) * expanded


def dequant_fp4_rows(weight: torch.Tensor, scale: torch.Tensor, cols: int) -> torch.Tensor:
    weight_u8 = weight.contiguous().view(torch.uint8)
    low = (weight_u8 & 0x0F).long()
    high = ((weight_u8 >> 4) & 0x0F).long()
    table = FP4_TABLE.to(weight_u8.device)
    values = torch.stack([table[low], table[high]], dim=-1).flatten(-2)
    scale_f32 = scale_to_f32(scale).repeat_interleave(FP4_BLOCK, dim=-1)
    return values[:, :cols] * scale_f32[:, :cols]


def finite_counts(tensor: torch.Tensor) -> int:
    if tensor.is_floating_point():
        return int((~torch.isfinite(tensor)).sum().item())
    return 0


def update_stats(stats: dict[str, Any], ref: torch.Tensor, got: torch.Tensor) -> None:
    ref_f = ref.to(torch.float32)
    got_f = got.to(torch.float32)
    stats["numel"] += ref_f.numel()
    stats["ref_nonfinite"] += finite_counts(ref_f)
    stats["got_nonfinite"] += finite_counts(got_f)

    finite = torch.isfinite(ref_f) & torch.isfinite(got_f)
    stats["finite_numel"] += int(finite.sum().item())
    if not finite.any():
        stats["max_abs_err"] = float("inf")
        return

    ref_valid = ref_f[finite]
    got_valid = got_f[finite]
    diff = (got_valid - ref_valid).abs()
    ref_abs = ref_valid.abs()
    stats["abs_err_sum"] += float(diff.sum().item())
    stats["ref_abs_sum"] += float(ref_abs.sum().item())
    stats["max_abs_err"] = max(stats["max_abs_err"], float(diff.max().item()))
    stats["max_ref_abs"] = max(stats["max_ref_abs"], float(ref_abs.max().item()))


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    finite_numel = max(int(stats["finite_numel"]), 1)
    mean_abs_err = float(stats["abs_err_sum"]) / finite_numel
    mean_ref_abs = float(stats["ref_abs_sum"]) / finite_numel
    rel_mean_err = mean_abs_err / max(mean_ref_abs, 1e-12)
    rel_max_err = float(stats["max_abs_err"]) / max(float(stats["max_ref_abs"]), 1e-12)
    stats.update(
        {
            "mean_abs_err": mean_abs_err,
            "mean_ref_abs": mean_ref_abs,
            "rel_mean_err": rel_mean_err,
            "rel_max_err": rel_max_err,
        }
    )
    return stats


def new_stats(key: str, input_dtype: str, quant_dtype: str, shape: tuple[int, ...]) -> dict[str, Any]:
    return {
        "key": key,
        "input_dtype": input_dtype,
        "quant_dtype": quant_dtype,
        "shape": list(shape),
        "numel": 0,
        "finite_numel": 0,
        "ref_nonfinite": 0,
        "got_nonfinite": 0,
        "abs_err_sum": 0.0,
        "ref_abs_sum": 0.0,
        "max_abs_err": 0.0,
        "max_ref_abs": 0.0,
    }


def compare_tensor(
    key: str,
    input_file: Any,
    quant_file: Any,
    quant_index: dict[str, str],
    quant_dir: Path,
    quant_files: dict[str, Any],
    row_chunk: int,
) -> dict[str, Any]:
    input_slice = input_file.get_slice(key)
    quant_slice = quant_file.get_slice(key)
    shape = tuple(input_slice.get_shape())
    input_dtype = input_slice.get_dtype()
    quant_dtype = quant_slice.get_dtype()
    stats = new_stats(key, input_dtype, quant_dtype, shape)

    if quant_dtype == "F8_E4M3":
        if len(shape) != 2:
            raise ValueError(f"FP8 tensor is not 2D: {key} shape={shape}")
        scale_key = key[: -len(".weight")] + ".scale"
        if scale_key not in quant_index:
            raise KeyError(f"Missing FP8 scale for {key}: {scale_key}")
        rows, cols = shape
        chunk = max(row_chunk, FP8_BLOCK)
        chunk = int(math.ceil(chunk / FP8_BLOCK) * FP8_BLOCK)
        scale_filename = quant_index[scale_key]
        scale_file = quant_files.get(scale_filename)
        for start in range(0, rows, chunk):
            end = min(start + chunk, rows)
            q = quant_slice[start:end]
            ref = input_slice[start:end]
            if scale_file is None:
                with safe_open(quant_dir / scale_filename, framework="pt", device="cpu") as scale_file:
                    scale = scale_file.get_slice(scale_key)[start // FP8_BLOCK : math.ceil(end / FP8_BLOCK)]
            else:
                scale = scale_file.get_slice(scale_key)[start // FP8_BLOCK : math.ceil(end / FP8_BLOCK)]
            got = dequant_fp8_rows(q, scale, start, end - start, cols)
            update_stats(stats, ref, got)
    elif quant_dtype == "I8":
        if len(shape) != 2:
            raise ValueError(f"FP4 tensor is not 2D: {key} shape={shape}")
        scale_key = key[: -len(".weight")] + ".scale"
        if scale_key not in quant_index:
            raise KeyError(f"Missing FP4 scale for {key}: {scale_key}")
        rows, cols = shape
        scale_filename = quant_index[scale_key]
        scale_file = quant_files.get(scale_filename)
        for start in range(0, rows, row_chunk):
            end = min(start + row_chunk, rows)
            q = quant_slice[start:end]
            ref = input_slice[start:end]
            if scale_file is None:
                with safe_open(quant_dir / scale_filename, framework="pt", device="cpu") as scale_file:
                    scale = scale_file.get_slice(scale_key)[start:end]
            else:
                scale = scale_file.get_slice(scale_key)[start:end]
            got = dequant_fp4_rows(q, scale, cols)
            update_stats(stats, ref, got)
    else:
        if len(shape) == 0:
            ref = input_file.get_tensor(key)
            got = quant_file.get_tensor(key)
            update_stats(stats, ref, got)
        elif len(shape) >= 1:
            rows = shape[0]
            for start in range(0, rows, row_chunk):
                end = min(start + row_chunk, rows)
                ref = input_slice[start:end]
                got = quant_slice[start:end]
                update_stats(stats, ref, got)

    return finalize_stats(stats)


def write_rank_report(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with tmp.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")
    tmp.rename(path)


def summarize(report_dir: Path, world_size: int, top_k: int, warn_rel_mean: float, warn_max_abs: float) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for rank in range(world_size):
        with (report_dir / f"rank_{rank}.jsonl").open() as f:
            records.extend(json.loads(line) for line in f if line.strip())
        error_path = report_dir / f"rank_{rank}.errors.jsonl"
        if error_path.exists():
            with error_path.open() as f:
                errors.extend(json.loads(line) for line in f if line.strip())

    suspicious = [
        r
        for r in records
        if r.get("got_nonfinite", 0) > 0
        or r.get("ref_nonfinite", 0) > 0
        or r.get("rel_mean_err", 0.0) > warn_rel_mean
        or r.get("max_abs_err", 0.0) > warn_max_abs
    ]
    summary = {
        "num_records": len(records),
        "num_errors": len(errors),
        "num_suspicious": len(suspicious),
        "thresholds": {"warn_rel_mean": warn_rel_mean, "warn_max_abs": warn_max_abs},
        "worst_by_rel_mean": sorted(records, key=lambda r: r.get("rel_mean_err", 0.0), reverse=True)[:top_k],
        "worst_by_max_abs": sorted(records, key=lambda r: r.get("max_abs_err", 0.0), reverse=True)[:top_k],
        "nonfinite": [
            r
            for r in records
            if r.get("got_nonfinite", 0) > 0 or r.get("ref_nonfinite", 0) > 0
        ][:top_k],
        "suspicious": sorted(
            suspicious,
            key=lambda r: (r.get("got_nonfinite", 0) + r.get("ref_nonfinite", 0), r.get("rel_mean_err", 0.0)),
            reverse=True,
        )[:top_k],
        "errors": errors[:top_k],
    }
    write_json_atomic(report_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError(f"Invalid rank/world_size: rank={args.rank}, world_size={args.world_size}")

    print(
        f"DeepSeek-V4 quant validation rank={args.rank}/{args.world_size} host={socket.gethostname()}",
        flush=True,
    )
    input_index = load_index(args.input_dir)
    quant_index = load_index(args.quant_dir)
    input_weight_map: dict[str, str] = input_index["weight_map"]
    quant_weight_map: dict[str, str] = quant_index["weight_map"]

    if args.rank == 0:
        if args.report_dir.exists():
            if args.overwrite:
                shutil.rmtree(args.report_dir)
            else:
                raise FileExistsError(f"Report dir already exists: {args.report_dir}")
        args.report_dir.mkdir(parents=True)
        args_payload = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        write_json_atomic(args.report_dir / "meta.json", args_payload | {"host": socket.gethostname()})
    else:
        wait_for_path(args.report_dir / "meta.json", args.barrier_timeout_seconds, "rank0 report init")

    by_input_shard = sorted(shard_to_keys(input_weight_map).items())
    assigned_shards = [
        (idx, filename, keys)
        for idx, (filename, keys) in enumerate(by_input_shard, start=1)
        if idx >= args.start_shard_index and (idx - 1) % args.world_size == args.rank
    ]
    if args.max_shards > 0:
        assigned_shards = assigned_shards[: args.max_shards]
    print(f"Rank {args.rank} assigned {len(assigned_shards)}/{len(by_input_shard)} shards", flush=True)

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for shard_idx, input_filename, keys in assigned_shards:
        print(f"[rank {args.rank} shard {shard_idx}/{len(by_input_shard)}] validating {input_filename}", flush=True)
        with safe_open(args.input_dir / input_filename, framework="pt", device="cpu") as input_file:
            output_files = sorted({quant_weight_map.get(k) for k in keys if k in quant_weight_map})
            with ExitStack() as stack:
                open_outputs = {
                    filename: stack.enter_context(safe_open(args.quant_dir / filename, framework="pt", device="cpu"))
                    for filename in output_files
                    if filename is not None
                }
                for key in keys:
                    try:
                        if key not in quant_weight_map:
                            raise KeyError(f"Missing quantized key: {key}")
                        quant_file = open_outputs[quant_weight_map[key]]
                        records.append(
                            compare_tensor(
                                key,
                                input_file,
                                quant_file,
                                quant_weight_map,
                                args.quant_dir,
                                open_outputs,
                                args.row_chunk,
                            )
                        )
                    except Exception as exc:
                        errors.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})

    write_rank_report(args.report_dir / f"rank_{args.rank}.jsonl", records)
    write_rank_report(args.report_dir / f"rank_{args.rank}.errors.jsonl", errors)
    write_json_atomic(args.report_dir / f"rank_{args.rank}.done.json", {"rank": args.rank, "records": len(records), "errors": len(errors)})

    if args.rank != 0:
        return

    deadline = time.time() + args.barrier_timeout_seconds
    pending = set(range(args.world_size))
    while pending:
        for rank in list(pending):
            if (args.report_dir / f"rank_{rank}.done.json").exists():
                pending.remove(rank)
        if pending:
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for ranks: {sorted(pending)}")
            time.sleep(10)

    summary = summarize(args.report_dir, args.world_size, args.top_k, args.warn_rel_mean, args.warn_max_abs)
    print(json.dumps(summary, indent=2, sort_keys=True)[:20000], flush=True)


if __name__ == "__main__":
    main()
