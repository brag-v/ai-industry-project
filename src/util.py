import matplotlib.pyplot as plt
import pandas as pd
import shap
import numpy as np

    
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

def calculate_shap_values(model, X_data, model_type='tree'):
    """
    Calculate SHAP values for model interpretability and feature interaction analysis.
    
    Args:
        model: Trained model (RandomForest or GradientBoosting)
        X_data: Feature data (pandas DataFrame)
        model_type: 'tree' for tree-based models
    
    Returns:
        explainer: SHAP explainer object
        shap_values: SHAP values array
    """
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, X_data)
    
    shap_values = explainer.shap_values(X_data)
    return explainer, shap_values

def plot_shap_summary(shap_values, X_data, model_name, plot_type="bar"):
    """
    Plot SHAP summary visualization showing feature importance and impact.
    
    Args:
        shap_values: SHAP values from explainer
        X_data: Feature data (pandas DataFrame)
        model_name: Name of the model for title
        plot_type: 'bar' for importance, 'dot' for interaction effects
    """
    plt.figure()
    shap.summary_plot(shap_values, X_data, plot_type=plot_type, show=False)
    plt.title(f"SHAP {plot_type.capitalize()} Plot - {model_name}")
    plt.tight_layout()
    plt.show()

def plot_shap_dependence(shap_values, X_data, feature_name, model_name):
    """
    Plot SHAP dependence plot to visualize feature interactions and effects.
    
    Args:
        shap_values: SHAP values from explainer
        X_data: Feature data (pandas DataFrame)
        feature_name: Name of feature to analyze
        model_name: Name of the model for title
    """
    plt.figure()
    shap.dependence_plot(feature_name, shap_values, X_data, show=False)
    plt.title(f"SHAP Dependence Plot: {feature_name} - {model_name}")
    plt.tight_layout()
    plt.show()