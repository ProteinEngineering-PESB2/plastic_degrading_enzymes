import warnings
warnings.filterwarnings("ignore")

import os

path_encoders = "../../processed_dataset/"
path_save_results = "../../results_training_exploring/"

encoder_strategies = os.listdir(path_encoders)

for encoder in ["k_mers_3"]:
    print("Processing encoder: ", encoder)

    command = f"python training_model.py {path_encoders}{encoder}/encoder_data.csv {path_save_results} {encoder}"
    os.system(command)        
