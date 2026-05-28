#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AUTOMODEL_ROOT_DEFAULT=$(cd "${SCRIPT_DIR}/.." && pwd)

echo "========================================="
echo "Start DeepSeek-V4 checkpoint -> vLLM export"
echo "Job Name: ${FULL_JOB_NAME:-manual}"
echo "Pod Name: ${POD_NAME:-manual}"
echo "========================================="

set +u
source /root/.bashrc || true
set -u

PYTHON_BIN=${PYTHON_BIN:-python3}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}

AUTOMODEL_ROOT=${AUTOMODEL_ROOT:-${AUTOMODEL_ROOT_DEFAULT}}
REFERENCE_MODEL_DIR=${REFERENCE_MODEL_DIR:-}
MODEL_DIR=${MODEL_DIR:-}
CONSOLIDATED_DIR=${CONSOLIDATED_DIR:-${MODEL_DIR}/consolidated}
VLLM_DIR=${VLLM_DIR:-${MODEL_DIR}/consolidated-fp8-fp4-vllm}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6001}
NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-}}
RDZV_ID=${RDZV_ID:-${FULL_JOB_NAME:-dsv4_checkpoint_to_vllm}}
RDZV_READ_TIMEOUT=${RDZV_READ_TIMEOUT:-600}
RDZV_JOIN_TIMEOUT=${RDZV_JOIN_TIMEOUT:-1800}
RDZV_LAST_CALL_TIMEOUT=${RDZV_LAST_CALL_TIMEOUT:-120}

CONSOLIDATE_BACKEND=${CONSOLIDATE_BACKEND:-gloo}
CONSOLIDATE_NUM_THREADS=${CONSOLIDATE_NUM_THREADS:-2}
CONSOLIDATE_USE_CUDA=${CONSOLIDATE_USE_CUDA:-0}
BACKUP_HF_METADATA=${BACKUP_HF_METADATA:-1}
OVERWRITE_CONSOLIDATED=${OVERWRITE_CONSOLIDATED:-0}

EXPERT_QUANT=${EXPERT_QUANT:-fp4}
ROW_CHUNK=${ROW_CHUNK:-256}
OVERWRITE_QUANT=${OVERWRITE_QUANT:-0}
VALIDATE_QUANT_OUTPUT=${VALIDATE_QUANT_OUTPUT:-1}
WAIT_FOR_INPUT_SECONDS=${WAIT_FOR_INPUT_SECONDS:-0}
BARRIER_TIMEOUT_SECONDS=${BARRIER_TIMEOUT_SECONDS:-86400}
COPY_CHAT_TEMPLATE=${COPY_CHAT_TEMPLATE:-}

if [[ -n "${POD_NAME:-}" ]]; then
  if [[ -z "${NODE_RANK}" || ( "${NODE_RANK}" = "0" && "${POD_NAME}" =~ -worker-[0-9]+$ ) ]]; then
    if [[ "${POD_NAME}" =~ -master-([0-9]+)$ ]]; then
      NODE_RANK="${BASH_REMATCH[1]}"
    elif [[ "${POD_NAME}" =~ -worker-([0-9]+)$ ]]; then
      NODE_RANK="$((BASH_REMATCH[1] + 1))"
    fi
  fi
fi
NODE_RANK=${NODE_RANK:-0}
if [[ -n "${MASTER_POD_IP:-}" ]]; then
  MASTER_ADDR=${MASTER_POD_IP}
fi
export MASTER_ADDR MASTER_PORT NNODES NODE_RANK

if [[ -z "${MODEL_DIR}" || -z "${REFERENCE_MODEL_DIR}" ]]; then
  cat <<'USAGE'
Required environment variables:
  MODEL_DIR             AutoModel checkpoint model directory, for example <checkpoint>/epoch_1_step_49/model
  REFERENCE_MODEL_DIR   Released DeepSeek-V4-Flash HF checkpoint used as serving schema

Optional environment variables:
  AUTOMODEL_ROOT        AutoModel source root; defaults to this repository
  CONSOLIDATED_DIR      Output BF16 HF consolidated directory; defaults to ${MODEL_DIR}/consolidated
  VLLM_DIR              Output FP8/FP4 vLLM directory; defaults to ${MODEL_DIR}/consolidated-fp8-fp4-vllm
  EXPERT_QUANT          fp4 or fp8; defaults to fp4 for DeepSeek-V4-Flash
  OVERWRITE_CONSOLIDATED=1 and/or OVERWRITE_QUANT=1 to rebuild existing outputs
USAGE
  exit 2
fi

NODE_IP=$(hostname -i 2>/dev/null | tr ' ' '\n' | grep '^10\.' | head -1 || true)
if [[ -z "${NODE_IP}" ]]; then
  NODE_IP=$(hostname -i 2>/dev/null | awk '{print $1}' || true)
fi
if [[ -z "${NODE_IP}" ]]; then
  NODE_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
fi

SOCKET_IFNAME=""
if [[ -n "${NODE_IP}" ]]; then
  if command -v ip >/dev/null 2>&1; then
    SOCKET_IFNAME=$(ip -o -4 addr show | awk -v ip="${NODE_IP}" '$4 ~ ("^" ip "/") {print $2; exit}')
  elif [[ -x /sbin/ip ]]; then
    SOCKET_IFNAME=$(/sbin/ip -o -4 addr show | awk -v ip="${NODE_IP}" '$4 ~ ("^" ip "/") {print $2; exit}')
  fi
fi
if [[ -n "${SOCKET_IFNAME}" ]]; then
  export GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}"
fi

NO_PROXY_LIST="127.0.0.1,localhost,${MASTER_ADDR}"
if [[ -n "${NODE_IP}" ]]; then
  NO_PROXY_LIST="${NO_PROXY_LIST},${NODE_IP}"
fi
if [[ -n "${MASTER_POD_IP:-}" ]]; then
  NO_PROXY_LIST="${NO_PROXY_LIST},${MASTER_POD_IP}"
fi
export no_proxy="${NO_PROXY_LIST}"
export NO_PROXY="${NO_PROXY_LIST}"

if [[ ! -d "${AUTOMODEL_ROOT}" ]]; then
  echo "AUTOMODEL_ROOT ${AUTOMODEL_ROOT} does not exist."
  exit 1
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR ${MODEL_DIR} does not exist."
  exit 1
fi
if [[ ! -f "${REFERENCE_MODEL_DIR}/model.safetensors.index.json" ]]; then
  echo "REFERENCE_MODEL_DIR ${REFERENCE_MODEL_DIR} is missing model.safetensors.index.json."
  exit 1
fi
if [[ ! -f "${AUTOMODEL_ROOT}/tools/offline_hf_consolidation.py" ]]; then
  echo "Missing offline_hf_consolidation.py under ${AUTOMODEL_ROOT}/tools."
  exit 1
fi
if [[ ! -f "${AUTOMODEL_ROOT}/tools/quantize_deepseek_v4_flash_hf.py" ]]; then
  echo "Missing quantize_deepseek_v4_flash_hf.py under ${AUTOMODEL_ROOT}/tools."
  exit 1
fi

export PYTHONPATH="${AUTOMODEL_ROOT}:${PYTHONPATH:-}"
if [[ "${CONSOLIDATE_USE_CUDA}" != "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

is_rank0=0
if [[ "${NODE_RANK}" == "0" ]]; then
  is_rank0=1
fi

consolidated_complete() {
  [[ -f "${CONSOLIDATED_DIR}/model.safetensors.index.json" ]] &&
    [[ -f "${CONSOLIDATED_DIR}/config.json" ]] &&
    [[ -f "${CONSOLIDATED_DIR}/tokenizer.json" ]] &&
    [[ -f "${CONSOLIDATED_DIR}/tokenizer_config.json" ]]
}

vllm_complete() {
  [[ -f "${VLLM_DIR}/model.safetensors.index.json" ]] &&
    [[ -f "${VLLM_DIR}/config.json" ]] &&
    [[ -f "${VLLM_DIR}/tokenizer.json" ]] &&
    [[ -f "${VLLM_DIR}/tokenizer_config.json" ]] &&
    [[ -f "${VLLM_DIR}/generation_config.json" ]] &&
    [[ -f "${VLLM_DIR}/encoding/encoding_dsv4.py" ]]
}

wait_for_vllm_complete() {
  local deadline=$((SECONDS + BARRIER_TIMEOUT_SECONDS))
  while ! vllm_complete; do
    if (( SECONDS > deadline )); then
      echo "Timed out waiting for vLLM output to become complete: ${VLLM_DIR}"
      return 1
    fi
    sleep 10
  done
}

echo "AUTOMODEL_ROOT: ${AUTOMODEL_ROOT}"
echo "REFERENCE_MODEL_DIR: ${REFERENCE_MODEL_DIR}"
echo "MODEL_DIR: ${MODEL_DIR}"
echo "CONSOLIDATED_DIR: ${CONSOLIDATED_DIR}"
echo "VLLM_DIR: ${VLLM_DIR}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"

cd "${AUTOMODEL_ROOT}"

if consolidated_complete && [[ "${OVERWRITE_CONSOLIDATED}" != "1" && "${OVERWRITE_CONSOLIDATED}" != "true" ]]; then
  echo "[1/3] Consolidated checkpoint already complete; skipping: ${CONSOLIDATED_DIR}"
else
  echo "[1/3] Consolidating sharded checkpoint to HF format..."

  HF_METADATA_DIR="${MODEL_DIR}/.hf_metadata"
  HF_METADATA_BACKUP_DIR="${MODEL_DIR}/.hf_metadata.backup"
  PREPARE_TOKEN=$(echo "${RDZV_ID}" | tr -c '[:alnum:]_.-' '_')
  PREPARE_DONE="${MODEL_DIR}/.checkpoint_to_vllm_prepare_done_${PREPARE_TOKEN}"

  if [[ "${is_rank0}" == "1" ]]; then
    rm -f "${PREPARE_DONE}"
    if [[ ! -d "${HF_METADATA_DIR}" && -d "${HF_METADATA_BACKUP_DIR}" ]]; then
      echo "Restoring .hf_metadata from backup: ${HF_METADATA_BACKUP_DIR}"
      cp -a "${HF_METADATA_BACKUP_DIR}" "${HF_METADATA_DIR}"
    fi
    if [[ ! -f "${HF_METADATA_DIR}/fqn_to_file_index_mapping.json" ]]; then
      echo "Missing ${HF_METADATA_DIR}/fqn_to_file_index_mapping.json; cannot consolidate."
      exit 1
    fi
    if [[ "${BACKUP_HF_METADATA}" == "1" && ! -d "${HF_METADATA_BACKUP_DIR}" ]]; then
      echo "Backing up .hf_metadata to ${HF_METADATA_BACKUP_DIR}"
      cp -a "${HF_METADATA_DIR}" "${HF_METADATA_BACKUP_DIR}"
    fi
    if [[ -d "${CONSOLIDATED_DIR}" && -n "$(find "${CONSOLIDATED_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      if [[ "${OVERWRITE_CONSOLIDATED}" == "1" || "${OVERWRITE_CONSOLIDATED}" == "true" ]]; then
        rm -rf "${CONSOLIDATED_DIR}"
      else
        echo "CONSOLIDATED_DIR exists but is incomplete: ${CONSOLIDATED_DIR}"
        exit 1
      fi
    fi
    mkdir -p "${CONSOLIDATED_DIR}"
    touch "${PREPARE_DONE}"
  else
    for _ in $(seq 1 600); do
      [[ -f "${PREPARE_DONE}" ]] && break
      sleep 1
    done
    if [[ ! -f "${PREPARE_DONE}" ]]; then
      echo "Timed out waiting for rank 0 to prepare ${CONSOLIDATED_DIR}."
      exit 1
    fi
  fi

  TORCHRUN_ARGS=(--nproc-per-node="${NPROC_PER_NODE}")
  if [[ "${NNODES}" == "1" ]]; then
    TORCHRUN_ARGS+=(--standalone)
  else
    TORCHRUN_ARGS+=(
      --nnodes="${NNODES}"
      --node-rank="${NODE_RANK}"
      --rdzv_backend=c10d
      --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
      --rdzv_id="${RDZV_ID}"
      --rdzv_conf="read_timeout=${RDZV_READ_TIMEOUT},join_timeout=${RDZV_JOIN_TIMEOUT},last_call_timeout=${RDZV_LAST_CALL_TIMEOUT}"
    )
  fi

  "${TORCHRUN_BIN}" "${TORCHRUN_ARGS[@]}" \
    tools/offline_hf_consolidation.py \
    --backend "${CONSOLIDATE_BACKEND}" \
    --model-name "${REFERENCE_MODEL_DIR}" \
    --input-dir "${MODEL_DIR}" \
    --output-dir "${CONSOLIDATED_DIR}" \
    --num-threads "${CONSOLIDATE_NUM_THREADS}"

  if ! consolidated_complete; then
    echo "Consolidation finished but output is incomplete: ${CONSOLIDATED_DIR}"
    exit 1
  fi
fi

if vllm_complete && [[ "${OVERWRITE_QUANT}" != "1" && "${OVERWRITE_QUANT}" != "true" ]]; then
  echo "[2/3] vLLM quantized checkpoint already complete; skipping: ${VLLM_DIR}"
else
  echo "[2/3] Exporting consolidated checkpoint to FP8/FP4 vLLM layout..."
  QUANT_ARGS=(
    --input-dir "${CONSOLIDATED_DIR}"
    --reference-model-dir "${REFERENCE_MODEL_DIR}"
    --output-dir "${VLLM_DIR}"
    --expert-quant "${EXPERT_QUANT}"
    --row-chunk "${ROW_CHUNK}"
    --rank "${NODE_RANK}"
    --world-size "${NNODES}"
    --barrier-timeout-seconds "${BARRIER_TIMEOUT_SECONDS}"
    --wait-for-input-seconds "${WAIT_FOR_INPUT_SECONDS}"
  )
  if [[ "${OVERWRITE_QUANT}" == "1" || "${OVERWRITE_QUANT}" == "true" ]]; then
    QUANT_ARGS+=(--overwrite)
  fi
  if [[ "${VALIDATE_QUANT_OUTPUT}" == "1" || "${VALIDATE_QUANT_OUTPUT}" == "true" ]]; then
    QUANT_ARGS+=(--validate-output)
  fi
  if [[ -n "${COPY_CHAT_TEMPLATE}" ]]; then
    QUANT_ARGS+=(--copy-chat-template "${COPY_CHAT_TEMPLATE}")
  fi

  "${PYTHON_BIN}" tools/quantize_deepseek_v4_flash_hf.py "${QUANT_ARGS[@]}"

  # Non-zero ranks return from the quantizer after writing their assigned
  # shards. Rank 0 then merges the index, writes metadata, validates the temp
  # directory, and renames it into VLLM_DIR. Do not let worker ranks fail the
  # job before rank 0 completes that finalization.
  wait_for_vllm_complete
fi

if [[ "${is_rank0}" == "1" ]]; then
  echo "[3/3] Verifying serving metadata..."
  "${PYTHON_BIN}" - "${VLLM_DIR}" "${EXPERT_QUANT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expert_quant = sys.argv[2]
cfg = json.loads((root / "config.json").read_text())
idx = json.loads((root / "model.safetensors.index.json").read_text())
keys = idx.get("weight_map", {})
required = {
    "model_type": "deepseek_v4",
    "torch_dtype": "bfloat16",
    "expert_dtype": expert_quant,
    "num_nextn_predict_layers": 0,
}
for key, expected in required.items():
    got = cfg.get(key)
    if got != expected:
        raise SystemExit(f"Bad config {key}: expected {expected!r}, got {got!r}")
if not cfg.get("quantization_config"):
    raise SystemExit("Missing quantization_config")
if not cfg.get("rope_scaling"):
    raise SystemExit("Missing rope_scaling")
scale_keys = sum(1 for key in keys if key.endswith(".scale") or key.endswith(".weight_scale_inv"))
if scale_keys <= 0:
    raise SystemExit("No quant scale tensors found in model.safetensors.index.json")
print(f"Verified {root}: keys={len(keys)} scale_keys={scale_keys}")
PY
fi

echo "DeepSeek-V4 vLLM export complete: ${VLLM_DIR}"
