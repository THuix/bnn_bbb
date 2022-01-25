#!/bin/bash
#SBATCH --job-name=900 # nom du job
#SBATCH --nodes=1	# number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10       
#SBATCH --time=01:00:00             # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=output_900.txt   # nom du fichier de sortie
#SBATCH --error=error_900.txt     # nom du fichier d'erreur (ici commun avec la sortie)
# nettoyage des modules charges en interactif et hérités par défaut
module purge
# chargement des modules
module load anaconda3/2020.02/gcc-9.2.0
source activate myenv
export PL_TORCH_DISTRIBUTED_BACKEND=gloo
srun python ../../exps/run_exp.py --dataset BOSTON --model_name Linear_regime_1 --range_N 900 --range_alpha 1. --nb_epochs 0 --project_name init_r1 --lr=0.001 --range_nb_samples 50 None
