#!/usr/bin/env bash
set -euo pipefail


#SBATCH --time 7:59:59
#SBATCH --account safer
#SBATCH --partition gpu
#SBATCH --gpus h100:1
#SBATCH --output="logs/%j.out"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 16G
#SBATCH --mail-user=parsa.rahimi@idiap.ch
#SBATCH --mail-type=END



MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
MODEL_REVISION="${MODEL_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
THINK="${THINK:-false}"
STORAGE_DTYPE="${STORAGE_DTYPE:-float32}"
SEED="${SEED:-0}"

# Tasks without much knowledge

RANDOM_PORT=$((10000 + RANDOM % 30000))
MAIN_PORT="${MAIN_PORT:-$RANDOM_PORT}"


for task in pope_random mme textvqa_val_lite ocrbench_v2 docvqa_val_lite;
do
echo "Running benchmark for $task ... "

accelerate launch --num_processes 1 --main_process_port "${MAIN_PORT}" -m lmms_eval \
    --model qwen3_5 \
    --model_args "pretrained=$MODEL,revision=$MODEL_REVISION,enable_thinking=$THINK"  \
    --gen_kwargs "max_new_tokens=$MAX_NEW_TOKENS,temperature=0.0" \
    --tasks "$task" \
    --log_samples \
    --output_path "./logs/analyzer-$MODEL" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --representation_dynamics \
    --representation_dynamics_output "./logs/representations-$MODEL/capability-$task/" \
    --representation_dynamics_pooling last_valid \
    --representation_dynamics_phases prefill \
    --representation_dynamics_storage_dtype "$STORAGE_DTYPE" \
    --limit "$NUM_SAMPLES"


echo "Waiting for few seconds ... "
sleep 5

done

# Knowledge-heavy benchmarks
for task in ok_vqa_val2014_lite medqa mmmu_val ai2d_lite gpqa_main_cot_zeroshot ;
do

echo "Running benchmark for $task ... "

RANDOM_PORT=$((10000 + RANDOM % 30000))
MAIN_PORT="${MAIN_PORT:-$RANDOM_PORT}"

accelerate launch --num_processes 1 --main_process_port "${MAIN_PORT}" -m lmms_eval \
    --model qwen3_5 \
    --model_args "pretrained=$MODEL,revision=$MODEL_REVISION,enable_thinking=$THINK"  \
    --gen_kwargs "max_new_tokens=$MAX_NEW_TOKENS,temperature=0.0" \
    --tasks "$task" \
    --log_samples \
    --output_path "./logs/analyzer-$MODEL" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --representation_dynamics \
    --representation_dynamics_output "./logs/representations-$MODEL/knowledge-$task/" \
    --representation_dynamics_pooling last_valid \
    --representation_dynamics_phases prefill \
    --representation_dynamics_storage_dtype "$STORAGE_DTYPE" \
    --limit "$NUM_SAMPLES"

echo "Waiting for few seconds ... "
sleep 5

done
