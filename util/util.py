import matplotlib.pyplot as plt
import pandas as pd

    
def load_data(data_file):
    data = pd.read_csv(data_file, sep=',', parse_dates=[2, 3])
    return data

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