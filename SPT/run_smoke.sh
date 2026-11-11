#!/bin/bash
#SBATCH --job-name=spt_llm_smoke
#SBATCH --partition=gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# SPT + LLM 조건 임베딩 스모크 테스트: 짧은 학습(main.py) → 추론 1회(inference.py)
# 제출: cd /gpfs/home1/mil090/uoscapstone/SPT/SPT && mkdir -p logs && sbatch run_smoke.sh
set -e
cd /gpfs/home1/mil090/uoscapstone/SPT/SPT
PY=/home1/mil090/miniconda3/envs/capstone/bin/python
export HF_HOME=/home1/mil090/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

COMMON="--dataset anuran --var 0.1 --d_token 4 --denoising_dim 1024 --lr1 0.001 \
        --batch_size1 512 --batch_size2 512 --max_beta 0.01"
EPOCHS1=${EPOCHS1:-50}
EPOCHS2=${EPOCHS2:-500}

nvidia-smi -L
echo "===== TRAIN (epochs=$EPOCHS1, epochs_2=$EPOCHS2) ====="
$PY main.py $COMMON --seed 0 --epochs $EPOCHS1 --epochs_2 $EPOCHS2

echo "===== INFERENCE: Hypsiboas ====="
$PY inference.py $COMMON --ver 0 --prompt "A frog call recording of genus Hypsiboas."

echo "===== INFERENCE: Adenomera ====="
$PY inference.py $COMMON --ver 0 --prompt "A frog call recording of genus Adenomera."
