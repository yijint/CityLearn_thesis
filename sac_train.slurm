#!/bin/bash

#SBATCH --job-name=sac  # create a short name for your job
#SBATCH --gres=gpu:1

#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)

#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=jt9744@princeton.edu # CHANGE THIS TO YOUR EMAIL 

module load anaconda3/2024.10
conda activate thesis_v2
python sac_train.py