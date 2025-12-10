import matplotlib.pyplot as plt
import pandas as pd
import shap
import numpy as np
import seaborn
import random

shap.initjs()
    
def load_data(data_file):
    data = pd.read_csv(data_file)
    return data

def cast_columns_to_categories(data):
    for col in ["road_type", "lighting", "weather", "time_of_day"]:
        data[col]=data[col].astype('category').cat.codes

def plot_dataframe(data, labels=None, vmin=-1.96, vmax=1.96,
                   figsize=None, s=4, xlabel=None, ylabel=None):
    plt.figure(figsize=figsize)
    plt.imshow(data.T.iloc[:, :], aspect='auto',
            cmap='RdBu', vmin=vmin, vmax=vmax)
    if labels is not None:
        # nonzero = data.index[labels != 0]
        ncol = len(data.columns)
        lvl = - 0.05 * ncol
        # plt.scatter(nonzero, lvl*np.ones(len(nonzero)),
        #         s=s, color='tab:orange')
        plt.scatter(labels.index, np.ones(len(labels)) * lvl,
                s=s,
                color=plt.get_cmap('tab10')(np.mod(labels, 10)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


def plot_inputs_against_target(input, target):
    scatter_plot_cols = ["curvature"]
    box_plot_cols = [c for c in input.columns if c not in scatter_plot_cols]

    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, col in enumerate(scatter_plot_cols):
        ax = axes[i]
        ax.scatter(input[col], target, alpha=0.1)
        ax.set_title(f"{col} vs accident_risk")

    for i, col in enumerate(box_plot_cols):
        ax = axes[i + len(scatter_plot_cols)]
        seaborn.boxplot(x=input[col], y=target, ax=ax)
        ax.set_title(f"{col} vs accident_risk")

    plt.tight_layout()
    plt.show()
    
def plot_input_distributions(df):
    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        ax = axes[i]
        data = df[col]

        # Handle booleans as integers
        if data.dtype == "bool":
            data = data.astype(int)

        # Decide bins
        if np.issubdtype(data.dtype, np.floating):
            bins = min(25, len(data.unique()))
            ax.hist(data, bins=bins, edgecolor='black')
        else:
            # Non-float → integer/categorical plot
            counts = data.value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values, edgecolor='black')
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index.astype(str))

        ax.set_title(col)

    plt.tight_layout()
    plt.show()


def all_pairs_dependence_plots(shap_values, X_data, features, model_name):
    fig, axes = plt.subplots(len(features), len(features) - 1, figsize=(20, 16))
    axes = axes.flatten()

    i = 0
    for feature in features:
        for interaction in features:
            if feature == interaction:
                continue
            ax = axes[i]
            i += 1
            shap.dependence_plot(
                feature,
                shap_values,
                X_data,
                show=False,
                interaction_index=interaction,
                ax=ax,
            )
            # ax.title(f"{feature} - {interaction} ({model_name})")
    plt.title(f"Shap feature dependencies ({model_name})")
    plt.tight_layout()
    plt.show()