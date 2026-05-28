#!/bin/bash
#SBATCH --partition=bw
#SBATCH --ntasks=1
#SBATCH --mem=1024
#SBATCH --time=00:05:00
#SBATCH --job-name=hello_pierce
#SBATCH --output=hello_%J_stdout.txt
#SBATCH --error=hello_%J_stderr.txt
#SBATCH --mail-type=ALL
# Note: add "#SBATCH --mail-user=your@email.com" once you know which email you want notifications at
#################################################

echo "=================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Job name:    $SLURM_JOB_NAME"
echo "Node:        $(hostname)"
echo "Date:        $(date)"
echo "Working dir: $(pwd)"
echo "=================================="

# Show what Python-related modules exist on Titan
echo "--- Searching for available Python modules ---"
module avail python 2>&1 | head -40
echo "=================================="

# Show currently loaded modules
echo "--- Modules currently loaded ---"
module list 2>&1
echo "=================================="

# Try the system python (may or may not exist without a module)
echo "--- Checking for python3 on default PATH ---"
which python3 && python3 --version || echo "python3 not on default PATH (will need module load)"
echo "=================================="

echo "Job complete."