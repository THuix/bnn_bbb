#!/bin/bash
#SBATCH --job-name=resnet_cifar10 # nom du job
#SBATCH --nodes=1	# number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10  
#SBATCH --hint=nomultithread     
#SBATCH --time=00:10:00             # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=output_resnet_cifar10.txt   # nom du fichier de sortie
#SBATCH --error=error_resnet_cifar10.txt     # nom du fichier d'erreur (ici commun avec la sortie)
# nettoyage des modules charges en interactif et hérités par défaut
module purge
# chargement des modules
module load anaconda3/2020.02/gcc-9.2.0
source activate myenv
export PL_TORCH_DISTRIBUTED_BACKEND=gloo
srun python ../exps/run_resnet.py --dataset CIFAR10 --range_alpha 0.000001 --nb_epochs 1 --project_name resnet_exp
