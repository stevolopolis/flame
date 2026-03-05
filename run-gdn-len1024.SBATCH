#!/bin/bash
# Usage: sbatch run.SBATCH <filename>

#SBATCH --account=torch_pr_36_mren
#SBATCH --time=04:00:00
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s"
#SBATCH --job-name=gdn1024
#SBATCH --output=slurm/logs/%A.out


module purge
module load "anaconda3/2025.06"

mkdir -p slurm/logs
mkdir -p slurm/scripts

scontrol write batch_script $SLURM_JOB_ID "slurm/scripts/$SLURM_JOB_ID.sh"

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS
export N_NODES=$SLURM_NNODES
export RANK=$SLURM_PROCID


source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh;
export PATH_TO_ENV=/scratch/sl12886/ttt/lact2
source activate /scratch/sl12886/ttt/lact2
export PATH=$PATH_TO_ENV/bin:$PATH;

seed=$1
echo "Seed: $seed"

# seed sweeps: [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
if [ -z "$2" ]; then
  lr=1e-3
else
  lr=$2
fi
echo "Learning rate: $lr"

export WANDB_NAME="gdn-bs512-seqlen1024-context1024-l4d256-lr$lr-seed$seed"
export DATA_FILES="/scratch/sl12886/data/long-data-collections-without-books/*.arrow"  # change to your data-path

config=configs/XS_gated_deltanet.json
dump_folder=exp/gdn/gdn-bs512-seqlen1024-context1024-l4d256/run0-lr$lr-seed$seed

if [ -z "$3" ]; then
  # Training
  echo "Running training..."

  NNODE=1 NGPU=1 LOG_RANK=0 bash train.sh \
    --job.config_file flame/models/fla.toml \
    --job.dump_folder $dump_folder \
    --model.config $config \
    --model.tokenizer_path fla-hub/transformer-1.3B-100B \
    --optimizer.name AdamW \
    --optimizer.eps 1e-15 \
    --optimizer.lr $lr \
    --lr_scheduler.warmup_steps 1024 \
    --lr_scheduler.lr_min 0.1 \
    --lr_scheduler.decay_type cosine \
    --training.batch_size 256 \
    --training.seq_len 1024 \
    --training.context_len 1024 \
    --training.gradient_accumulation_steps 1 \
    --activation_checkpoint.mode selective \
    --activation_checkpoint.selective_ac_option 2 \
    --training.steps 40960 \
    --training.max_norm 1.0 \
    --training.skip_nan_inf \
    --training.dataset arrow \
    --training.dataset_split train \
    --training.data_dir "/scratch/sl12886/data/long-data-collections-train" \
    --training.num_workers 2 \
    --training.prefetch_factor 1 \
    --training.seed $seed \
    --training.compile \
    --checkpoint.interval 4096 \
    --checkpoint.load_step -1 \
    --checkpoint.keep_latest_k 2 \
    --metrics.log_freq 10 \
    --profiling.profile_freq 2000
else 
  # Evaluation
  echo "Running evaluation..."

  export TORCH_COMPILE_DISABLE=1
  NNODE=1 NGPU=1 LOG_RANK=0 bash eval.sh \
    --job.config_file flame/models/fla.toml \
    --job.dump_folder $dump_folder \
    --model.config $config \
    --model.tokenizer_path fla-hub/transformer-1.3B-100B \
    --eval.batch_size 16 \
    --eval.seq_len 65536 \
    --eval.context_len 65536 \
    --eval.steps 128 \
    --eval.dataset arrow \
    --eval.dataset_split train \
    --eval.data_dir "/scratch/sl12886/data/long-data-collections-val" \
    --eval.num_workers 2 \
    --eval.prefetch_factor 1 \
    --eval.seed 42
fi
