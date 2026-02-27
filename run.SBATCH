# #!/bin/bash
# # Usage: sbatch run.SBATCH <filename>

# #SBATCH --account=torch_pr_36_mren
# #SBATCH --time=00:10:00
# #SBATCH --mem=64g
# #SBATCH --cpus-per-task=16
# #SBATCH --ntasks=1
# #SBATCH --gres=gpu:1
# #SBATCH --constraint="l40s"
# #SBATCH --job-name=lact-test
# #SBATCH --output=slurm/logs/%A.out


# module purge
# module load "anaconda3/2025.06"

# mkdir -p slurm/logs
# mkdir -p slurm/scripts

# scontrol write batch_script $SLURM_JOB_ID "slurm/scripts/$SLURM_JOB_ID.sh"

# # export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
# export MASTER_ADDR=$(hostname)
# export MASTER_PORT=12345
# export WORLD_SIZE=$SLURM_NTASKS
# export N_NODES=$SLURM_NNODES
# export RANK=$SLURM_PROCID


# source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh;
# export PATH_TO_ENV=/scratch/sl12886/ttt/lact
# source activate /scratch/sl12886/ttt/lact
# export PATH=$PATH_TO_ENV/bin:$PATH;


# export WANDB_NAME="lact-test"
# export DATA_FILES="/scratch/sl12886/data/long-data-collections-without-books/*.arrow"  # change to your data-path

NNODE=1 NGPU=1 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/lact/lact-swa2048-rope-fw02-id-rank32-init0.5-gain0.5-nh4-momentum-muon-760M-32K-40B/batch1.seqlen32768.bs32.warmup1024.update1.steps40960.lr1e-3.cosine.32gpu \
  --model.config configs/XS_lact_swiglu_nh4_fwlow_rank_momentum_muon.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 1e-3 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 1 \
  --training.seq_len 256 \
  --training.context_len 256 \
  --training.gradient_accumulation_steps 1 \
  --activation_checkpoint.mode selective \
  --activation_checkpoint.selective_ac_option 2 \
  --training.steps 16384 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset arrow \
  --training.dataset_split train \
  --training.data_dir "/scratch/sl12886/data/long-data-collections-test" \
  --training.num_workers 2 \
  --training.prefetch_factor 1 \
  --training.seed 42 \
  --training.compile \
  --checkpoint.interval 4096 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 10 \
  --profiling.profile_freq 2000 \