#!/bin/bash
#SBATCH --job-name=init_job # nom du job
#SBATCH --nodes=1	# number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10       
#SBATCH --time=0:45:00             # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=output_initt.txt   # nom du fichier de sortie
#SBATCH --error=error_init.txt     # nom du fichier d'erreur (ici commun avec la sortie)
# nettoyage des modules charges en interactif et hérités par défaut
module purge
# chargement des modules
module load anaconda3/2020.02/gcc-9.2.0
source activate myenv
export PL_TORCH_DISTRIBUTED_BACKEND=gloo
#srun python ../exps/run_exp.py --dataset MNIST --model_name Linear_regime_1 --range_N 100 200 300 400 500 600 700 800 900 1000 --range_alpha None --nb_epochs 0 --project_name init_all
#srun python ../exps/run_exp.py --dataset BOSTON --model_name Linear_regime_1 --range_N 100 200 300 400 500 600 700 800 900 1000 --range_alpha None --nb_epochs 0 --project_name init_all
srun python ../exps/run_exp.py --dataset MNIST --model_name Linear_regime_3 --range_N 100 200 300 400 500 600 700 800 900 1000 --range_alpha 0.0000167 --nb_epochs 0 --project_name init_all
srun python ../exps/run_exp.py --dataset BOSTON --model_name Linear_regime_3 --range_N 100 200 300 400 500 600 700 800 900 1000 --range_alpha 0.0000167 --nb_epochs 0 --project_name init_all
#srun python ../exps/run_exp.py --dataset MNIST --model_name Conv_regime_1 --range_N 100 200 300 400 500 600 700 800 900 1000 --range_alpha None --nb_epochs 0 --project_name init_all
#srun python ../exps/run_exp.py --dataset MNIST --model_name Conv_regime_3 --range_N 100 200 300 400 500 600 700 800 900 1000 --range_alpha 0.0000167 --nb_epochs 0 --project_name init_all
