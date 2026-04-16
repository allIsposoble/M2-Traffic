#!/usr/bin/env bash
set -euo pipefail

# Optional debug mode: DEBUG=1 ./scripts/run_iscx_app_3round.sh
if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

ROOT="${ROOT:-$(pwd)}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

VOCAB="${VOCAB:-$ROOT/pre-training/vocab.txt}"
TRAIN="${TRAIN:-$ROOT/data/finetune_data/ISCX_App_Task/dataset/train_balanced.tsv}"
DEV="${DEV:-$ROOT/data/finetune_data/ISCX_App_Task/dataset/valid_dataset.tsv}"
TEST="${TEST:-$ROOT/data/finetune_data/ISCX_App_Task/dataset/test_dataset.tsv}"
PT="${PT:-$ROOT/model.bin-90000}"
CONFIG="${CONFIG:-models/bert/base_config.json}"

mkdir -p "$ROOT/logs" "$ROOT/models"

for f in "$VOCAB" "$TRAIN" "$DEV" "$TEST" "$PT"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing required file: $f" >&2
    exit 1
  fi
done

COMMON_ARGS=(
  --vocab_path "$VOCAB"
  --train_path "$TRAIN"
  --dev_path "$DEV"
  --test_path "$TEST"
  --pretrained_model_path "$PT"
  --config_path "$CONFIG"
  --tokenizer bert
  --batch_size 32
  --seq_length 128
  --epochs_num 20
  --earlystop 5
  --warmup 0.1
  --report_steps 50
  --seed 7
)

echo "[Round 1/3] Baseline"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" fine-tuning/run_classifier.py \
  "${COMMON_ARGS[@]}" \
  --output_model_path models/iscx_app_r1_baseline.bin \
  --learning_rate 2e-5 \
  --pooling first | tee logs/iscx_app_r1.log

echo "[Round 2/3] TD + Hybrid pooling"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" fine-tuning/run_classifier.py \
  "${COMMON_ARGS[@]}" \
  --output_model_path models/iscx_app_r2_td_hybrid.bin \
  --learning_rate 2e-5 \
  --use_td_encoder \
  --td_kernel_sizes 3,5,7 \
  --td_alpha 1.0 \
  --td_dropout 0.1 \
  --use_hybrid_pooling \
  --pooling first | tee logs/iscx_app_r2.log

echo "[Round 3/3] TD + Hybrid + SCM-ArcFace"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" fine-tuning/run_classifier.py \
  "${COMMON_ARGS[@]}" \
  --output_model_path models/iscx_app_r3_td_hybrid_arcface.bin \
  --learning_rate 1e-5 \
  --use_td_encoder \
  --td_kernel_sizes 3,5,7 \
  --td_alpha 1.0 \
  --td_dropout 0.1 \
  --use_hybrid_pooling \
  --use_scm_arcface \
  --arcface_s 30 \
  --arcface_m_base 0.2 \
  --arcface_m_lambda 0.3 \
  --pce_hidden_size 128 \
  --pooling first | tee logs/iscx_app_r3.log

echo "[Done] Logs saved to logs/iscx_app_r*.log"
echo "[Done] Parsing logs into CSV..."
"$PYTHON_BIN" scripts/parse_finetune_logs_to_csv.py \
  --log_glob "logs/iscx_app_r*.log" \
  --output_csv "logs/iscx_app_record_auto.csv"

echo "[Done] CSV: logs/iscx_app_record_auto.csv"
