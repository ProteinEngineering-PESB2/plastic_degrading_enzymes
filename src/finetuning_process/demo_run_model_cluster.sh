#!/bin/bash
#$ -cwd
#$ -N fine-tuning-ESM-V1
#$ -q GPU
#$ -l slotsGPU_1=1
#$ -l slotsGPU_0=0
#$ -l memGPU=32
#$ -pe parallelGPU 1

export PATH=$HOME/miniconda3/bin:$PATH

source activate updated_bioembedding 
python3 /home/dmedina/Desktop/projects/plastic_degrading_enzymes/src/finetuning_process/fine_tuning_esm.py /home/dmedina/Desktop/projects/plastic_degrading_enzymes/results/generated_dataset/PLA/processed_data.csv 0.2 42 /home/dmedina/Desktop/projects/plastic_degrading_enzymes/results/training_models/PLA/ facebook/esm2_t36_3B_UR50D