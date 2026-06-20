#!/bin/bash
#SBATCH --partition=bw
#SBATCH --ntasks=1
#SBATCH --mem=2048
#SBATCH --time=00:10:00
#SBATCH --job-name=cr3bp_test
#SBATCH --output=cr3bp_test_%J_stdout.txt
#SBATCH --error=cr3bp_test_%J_stderr.txt

module load Python/3.12.3-GCCcore-13.3.0
source ~/qpit-project/venv/bin/activate

cd ~/qpit-project
python3 titan_cr3bp_test.py