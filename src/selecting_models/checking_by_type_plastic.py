import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import sys
from model_selection import ModelSelection
from ploting_figures import MakePlots

df_data = pd.read_csv(sys.argv[1])
df_data["random_seed"] = df_data.index
path_results = sys.argv[2]

model_selection = ModelSelection(df_data=df_data)

index=16
while True:
    print("Testing with index: ", index)
    selected_methods = model_selection.select(min_votes_number=index)
    
    if len(selected_methods)>0:
        selected_methods.to_csv(f"{path_results}selected_models_for_tuning.csv", index=False)
        break
    else:
        index-=1

print("Plotting data")
model_selection.df_process.to_csv(f"{path_results}processed_performances.csv", index=False)

make_plots = MakePlots(dataset=model_selection.df_process, path_export=path_results)
make_plots.plot_by_algorithm()
make_plots.plot_by_encoder()
make_plots.plot_by_type_encoder()
make_plots.plot_filter_by_nlp()
