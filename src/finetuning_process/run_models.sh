#!/user/bin/bash

for plastic in 'NYLON_PA'  'PBAT'  'PCL'  'PET'  'PHA'  'PHB'  'PLA'  'PU_PUR'
do
for model in 
'facebook/esm2_t6_8M_UR50D'
'facebook/esm2_t12_35M_UR50D'
'facebook/esm2_t36_3B_UR50D'
'facebook/esm1b_t33_650M_UR50S'
'Rostlab/prot_t5_xl_uniref50'
'Rostlab/prot_t5_xl_bfd'
'ElnaggarLab/ankh2-ext1'
'ElnaggarLab/ankh2-ext2'
'ElnaggarLab/ankh2-large'
'RaphaelMourad/Mistral-Prot-v1-134M'
'RaphaelMourad/Mistral-Prot-v1-15M' 'RaphaelMourad/Mistral-Prot-v1-417M' 
'RaphaelMourad/Mistral-Peptide-v1-15M' 'RaphaelMourad/Mistral-Peptide-v1-134M' 'RaphaelMourad/Mistral-Peptide-v1-422M'
do 
echo fine_tuning_esm.py ../../results/generated_dataset/$plastic/processed_data.csv 0.2 42 ../../results/training_models/$model/
done
done
