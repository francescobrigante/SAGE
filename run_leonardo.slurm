#!/bin/bash

#SBATCH --job-name=cvae_train          # Nome del job (visibile in squeue e log)
#SBATCH --account=IscrC_AHNetBio       # Account di progetto su cui scalare le ore
#SBATCH --partition=boost_usr_prod     # Partizione GPU A100
#SBATCH --qos=boost_qos_lprod          # QoS lprod ottimale per training lunghi (max 96h). dbg per script di debugging < 30m.
#SBATCH --nodes=1                      # Numero di nodi richiesti
#SBATCH --ntasks-per-node=2            # Processi per nodo. Lightning in DDP necessita di 1 task dedicato per ogni GPU usata.
#SBATCH --gres=gpu:2                   # Numero di GPU A100 richieste per nodo (max 4 per nodo Booster)
#SBATCH --cpus-per-gpu=8               # Core CPU (thread) allocati per GPU, vitali per evitare colli di bottiglia sui Dataloader
#SBATCH --mem-per-gpu=128G             # RAM assegnata per GPU (tot max 512 sul nodo). 128G è 1/4 esatto.
#SBATCH --time=2-00:00:00              # Walltime massimo allocato (Giorni-Ore:Min:Sec)
#SBATCH --error=logs/%x_%j.err         # Routing target log stderr
#SBATCH --output=logs/%x_%j.out        # Routing target log stdout

module purge                           # Azzera env login per prevenire conflitti software CUDA/NVHPC
module load profile/deeplrn            # Espone il core tooling AI module di Leonardo
module load cuda/12.2 nvhpc            # Fixa i runtime paths di NVHPC verso cuda

export CUDA=$CUDA_HOME                 # Patch per local compilation path
export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU # Vincola lib numeriche ai core allocati, impedendo stallo deadlock CPU

export NCCL_DEBUG=INFO                          # Attiva warning network tra i worker Multi-GPU/DDP per check instradamento
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1        # Cruciale: fa fallire intieramente il job se una rank si corrompe (invece di freezare succhiando preziose ore core)

export WANDB_MODE=offline                       # Forza logging su disco (compute non ha inet). Sincronizzare dopo da login node con `wandb sync`
export WANDB_TAGS="$SLURM_JOB_NAME"             # Passa l'id test automaticamente a dashboard Wandb cloud

cd $WORK/C-VAE                         # Assicurarsi esplicito path prima di runnare sorgente
source .envs/c_vae/bin/activate        # Attiva ambiente target project C-VAE 

srun python train.py experiment=cplx trainer.devices=$SLURM_NTASKS_PER_NODE     # srun delega nativamente DDP topology a pl.Trainer