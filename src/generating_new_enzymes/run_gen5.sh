#!/bin/bash
# 
#$ -cwd
#$ -N generating_5
#$ -q GPU
#$ -l slotsGPU_1=0
#$ -l slotsGPU_0=1
#$ -l memGPU=24
#$ -pe parallelGPU 1

module load cuda/12.2
source /raid/data/dmedina/miniconda3/etc/profile.d/conda.sh
conda activate protein_engineering

sh /home/dmedina/Desktop/projects/plastic_degrading_enzymes/src/generating_new_enzymes/run_generation_enzymes.sh 3.5.2.12 0
