#!/bin/bash
#SBATCH --job-name=100 # nom du job
#SBATCH --nodes=1	# number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10       
#SBATCH --time=05:00:00             # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=output_100.txt   # nom du fichier de sortie
#SBATCH --error=error_100.txt     # nom du fichier d'erreur (ici commun avec la sortie)
# nettoyage des modules charges en interactif et hérités par défaut
module purge
# chargement des modules
module load anaconda3/2020.02/gcc-9.2.0
source activate myenv
export PL_TORCH_DISTRIBUTED_BACKEND=gloo
srun python ../../exps/run_exp.py --dataset CIFAR10 --model_name Resnet_regime_3 --range_N 1000 --range_alpha 100. --nb_epochs 200 --project_name resnet_new --lr=0.001 --range_nb_samples 1 None
