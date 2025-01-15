import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
sns.set_style("whitegrid")
plt.rc('font', size=12)

class MakePlots(object):

    def __init__(
            self, 
            dataset=None, 
            path_export=""):
        
        self.palette_values = ['#00B0BE', '#F45F74']
        self.colors = sns.color_palette(self.palette_values)

        self.dataset = dataset
        self.path_export = path_export
    
    def plot_by_type_encoder(self, name_fig="ml_classic_performance_by_type_encoder.png"):

        fig = plt.figure(figsize=(13,13))
        gs = GridSpec(2, 2, figure=fig)

        ax_data = fig.add_subplot(gs[0, 0])
        sns.boxplot(ax=ax_data, data=self.dataset, y="Strategy", x="Accuracy", hue="Stage", fill=False, palette=self.colors)
        
        ax_data = fig.add_subplot(gs[0, 1])
        sns.boxplot(ax=ax_data, data=self.dataset, y="Strategy", x="Precision", hue="Stage", fill=False, palette=self.colors)
        
        ax_data = fig.add_subplot(gs[1, 0])
        sns.boxplot(ax=ax_data, data=self.dataset, y="Strategy", x="Recall", hue="Stage", fill=False, palette=self.colors)
        
        ax_data = fig.add_subplot(gs[1, 1])
        sns.boxplot(ax=ax_data, data=self.dataset, y="Strategy", x="F1", hue="Stage", fill=False, palette=self.colors)

        plt.tight_layout()

        plt.savefig(f"{self.path_export}{name_fig}", dpi=300)
    
    def plot_by_encoder(self, name_fig="ml_classic_performance_by_encoder.png"):
        
        fig = plt.figure(figsize=(13,13))
        gs = GridSpec(2, 2, figure=fig)

        ax_data1 = fig.add_subplot(gs[0, 0])
        ax_data2 = fig.add_subplot(gs[0, 1])
        ax_data3 = fig.add_subplot(gs[1, 0])
        ax_data4 = fig.add_subplot(gs[1, 1])

        sns.boxplot(ax=ax_data1, data=self.dataset, y="Encoder", x="Accuracy", hue="Stage", fill=False, palette=self.colors)
        sns.boxplot(ax=ax_data2, data=self.dataset, y="Encoder", x="Precision", hue="Stage", fill=False, palette=self.colors)
        sns.boxplot(ax=ax_data3, data=self.dataset, y="Encoder", x="Recall", hue="Stage", fill=False, palette=self.colors)
        sns.boxplot(ax=ax_data4, data=self.dataset, y="Encoder", x="F1", hue="Stage", fill=False, palette=self.colors)

        plt.tight_layout()
        plt.savefig(f"{self.path_export}{name_fig}", dpi=300)
    
    def plot_by_algorithm(self, name_fig="ml_classic_performance_by_algorithm.png"):
        
        fig = plt.figure(figsize=(13,13))
        gs = GridSpec(2, 2, figure=fig)

        ax_data1 = fig.add_subplot(gs[0, 0])
        ax_data2 = fig.add_subplot(gs[0, 1])
        ax_data3 = fig.add_subplot(gs[1, 0])
        ax_data4 = fig.add_subplot(gs[1, 1])

        sns.boxplot(ax=ax_data1, data=self.dataset, y="Algorithm", x="Accuracy", hue="Stage", fill=False, palette=self.colors)
        sns.boxplot(ax=ax_data2, data=self.dataset, y="Algorithm", x="Precision", hue="Stage", fill=False, palette=self.colors)
        sns.boxplot(ax=ax_data3, data=self.dataset, y="Algorithm", x="Recall", hue="Stage", fill=False, palette=self.colors)
        sns.boxplot(ax=ax_data4, data=self.dataset, y="Algorithm", x="F1", hue="Stage", fill=False, palette=self.colors)

        plt.tight_layout()
        plt.savefig(f"{self.path_export}{name_fig}", dpi=300)
    
    def plot_filter_by_nlp(self, name_fig="ml_classic_performance_filter_NLP_by_type_encoder.png"):

        df_filter = self.dataset[self.dataset["Strategy"] == "Embedding-based"]

        fig = plt.figure(figsize=(13,13))
        gs = GridSpec(2, 2, figure=fig)

        ax_data1 = fig.add_subplot(gs[0, 0])
        ax_data2 = fig.add_subplot(gs[0, 1])
        ax_data3 = fig.add_subplot(gs[1, 0])
        ax_data4 = fig.add_subplot(gs[1, 1])

        sns.boxplot(ax=ax_data1, data=df_filter, y="Encoder", x="Accuracy", hue="Stage", fill=False, palette=self.colors)
        sns.boxplot(ax=ax_data2, data=df_filter, y="Encoder", x="Precision", hue="Stage", fill=False, palette=self.colors)
        sns.boxplot(ax=ax_data3, data=df_filter, y="Encoder", x="Recall", hue="Stage", fill=False, palette=self.colors)
        sns.boxplot(ax=ax_data4, data=df_filter, y="Encoder", x="F1", hue="Stage", fill=False, palette=self.colors)

        plt.tight_layout()
        plt.savefig(f"{self.path_export}{name_fig}", dpi=300)