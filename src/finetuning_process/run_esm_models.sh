#!/usr/bin/bash

for plastic in 'NYLON_PA'  'PBAT'  'PCL'  'PET'  'PHA'  'PHB'  'PLA'  'PU_PUR'
do
for model in 'esm2_t6_8M_UR50D' 'esm2_t12_35M_UR50D' 'esm2_t36_3B_UR50D' 'esm1b_t33_650M_UR50S'
do
python3 /home/dmedina/Desktop/projects/plastic_degrading_enzymes/src/finetuning_process/fine_tuning_esm.py -d /scratch/global_1/dmedina/results_generating_enzymes/results/generated_dataset/$plastic/processed_data.csv -o /scratch/global_1/dmedina/results_generating_enzymes/results/training_models/$plastic/ -m $model
done
done