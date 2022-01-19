#!/bin/bash
#SBATCH --job-name=init_p_scales # nom du job
#SBATCH --nodes=1	# number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10       
#SBATCH --time=0:45:00             # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=output_init_p_scales.txt   # nom du fichier de sortie
#SBATCH --error=error_init_p_scales.txt     # nom du fichier d'erreur (ici commun avec la sortie)
# nettoyage des modules charges en interactif et hérités par défaut
module purge
# chargement des modules
module load anaconda3/2020.02/gcc-9.2.0
source activate myenv
export PL_TORCH_DISTRIBUTED_BACKEND=gloo
srun python ../exps/run_exp.py --dataset MNIST --model_name Linear_regime_1 --range_N 50 150 250 350 450 550 650 750 850 950 --range_alpha None --nb_epochs 0 --project_name init_p_scales --p_scales_with_N True
