#!/bin/bash
#SBATCH --job-name=train_conv_regime_3_cifar10 # nom du job
#SBATCH --nodes=1	# number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10       
#SBATCH --time=24:00:00             # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=output_train_conv_regime_3_cifar10.txt   # nom du fichier de sortie
#SBATCH --error=error_train_conv_regime_3cifar10.txt     # nom du fichier d'erreur (ici commun avec la sortie)
# nettoyage des modules charges en interactif et hérités par défaut
module purge
# chargement des modules
module load anaconda3/2020.02/gcc-9.2.0
source activate myenv
export PL_TORCH_DISTRIBUTED_BACKEND=gloo
srun python ../exps/run_exp.py --dataset CIFAR10 --model_name Conv_regime_3 --range_N 100 --range_alpha 0.0001 0.001 0.01 0.1 1. 10. 100. --nb_epochs 100 --project_name train_exp
