#!/bin/bash
#SBATCH --job-name=resnet_classic_cifar # nom du job
#SBATCH --nodes=1	# number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10       
#SBATCH --time=5:00:00             # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=output_resnet_classic.txt   # nom du fichier de sortie
#SBATCH --error=error_resnet_classic.txt     # nom du fichier d'erreur (ici commun avec la sortie)
# nettoyage des modules charges en interactif et hérités par défaut
module purge
# chargement des modules
module load anaconda3/2020.02/gcc-9.2.0
source activate myenv
export PL_TORCH_DISTRIBUTED_BACKEND=gloo
srun python ../../exps/run_exp.py --dataset CIFAR10 --model_name Resnet20_classic --range_N 100 --range_alpha None --nb_epochs 100 --project_name resnet --lr=0.001 None
