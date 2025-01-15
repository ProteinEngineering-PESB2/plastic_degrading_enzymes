import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from classification_models import ClassificationModel

df_data = pd.read_csv(sys.argv[1])
path_export = sys.argv[2]
name_encoder = sys.argv[3]

df_responses = pd.read_csv("../../raw_data/enzymes_plastics.csv")
seeds_values = pd.read_csv("seeds.csv")

### Processing by plastic
for type_plastic in ["PET", "PHB", "PHA", "PLA", "PCL", "PU/PUR", "NYLON/PA"]:

    print("Processing plastic: ", type_plastic)
    
    ### create folder by type of plastic
    path_save_exploration = f"{path_export}{type_plastic}/{name_encoder}" if "/" not in type_plastic else f"{path_export}{type_plastic.replace("/", "_")}/{name_encoder}"
    
    command = f"mkdir -p {path_save_exploration}"
    os.system(command)

    ### processing dataset
    df_to_use = df_responses[[type_plastic, "sequence"]]
    df_merge = df_data.merge(right=df_to_use, on="sequence")
    df_merge = df_merge.drop(columns=["sequence"])
    df_merge = df_merge.dropna()

    ### generating positive/negative datasets
    positive_dataset = df_merge[df_merge[type_plastic] == 1]
    negative_dataset = df_merge[df_merge[type_plastic] == 0]

    print("Start training by applying seeds")
    for seed in [42, 2500, 7714, 9104, 5767, 2904, 1540, 9118, 1520, 5785]:
        extracted_negative = shuffle(negative_dataset, random_state=seed, n_samples=len(positive_dataset))

        df_to_train = pd.concat([positive_dataset, extracted_negative], axis=0)
        response_values = df_to_train[type_plastic]
        df_values = df_to_train.drop(columns=[type_plastic])

        X_train, X_val, y_train, y_val = train_test_split(df_values.values, response_values, test_size=.20, random_state=seed, shuffle=True)

        class_models = ClassificationModel(
            train_response=y_train,
            train_values=X_train,
            test_response=y_val,
            test_values=X_val
        )

        name_export = f"{path_save_exploration}/exploring_process_{seed}.csv"
        df_exploration = class_models.apply_exploring()
        df_exploration.to_csv(name_export, index=False)