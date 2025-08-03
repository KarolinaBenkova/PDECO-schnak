#!/bin/sh
#$ -N SV_PDECO                 # Job name
#$ -cwd                        # Run from the current working directory
#$ -l h_rt=20:00:00            # Runtime limit (hh:mm:ss)
#$ -l h_vmem=15G               # Memory limit per core

# Load environment modules
. /etc/profile.d/modules.sh

# Load Anaconda and activate Python environment
module load anaconda
source activate pythonfenicsenv

# Print CPU model for reference (optional)
cat /proc/cpuinfo | grep "model name" | head -1

# Execute Python script of choice
python SV_Neumann_ex.py

