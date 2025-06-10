import matplotlib.pyplot as plt
import numpy as np

# Define models and metrics
models = ["Baseline", "LSTM", "RNN", "LogFormer"]
metrics = ["Accuracy", "Top-5 Accuracy", "BLEU"]

# Dataset values
dataset1 = [
    [0.8519, 0.9887, 0.6989],
    [0.8385, 0.9765, 0.6869],
    [0.8418, 0.9784, 0.6889],
    [0.8992, 0.9951, 0.7816],
]

dataset2 = [
    [0.9442, 0.9843, 0.8992],
    [0.9815, 0.9996, 0.9572],
    [0.9806, 0.9996, 0.9555],
    [0.9884, 0.9999, 0.9718],
]

dataset3 = [
    [0.8417, 0.9997, 0.6538],
    [0.8431, 0.9997, 0.6562],
    [0.8429, 0.9997, 0.6558],
    [0.8898, 0.9999, 0.7496],
]

# Group datasets
datasets = [dataset1, dataset2, dataset3]
dataset_names = ["XC40", "Windows", "HDFS"]

# Plotting
x = np.arange(len(models))
width = 0.2

fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, (ax, data, name) in enumerate(zip(axs, datasets, dataset_names)):
    data = np.array(data).T  # Transpose to access each metric
    for j, (metric, color) in enumerate(zip(metrics, ['tab:blue', 'tab:orange', 'tab:green'])):
        ax.bar(x + j * width - width, data[j], width, label=metric if i == 0 else "", color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.set_title(name)
    ax.set_ylim([0.6, 1.05])
    ax.grid(True, axis='y')

axs[0].set_ylabel("Score")
fig.legend(loc='upper center', ncol=3)
#fig.suptitle("Model Performance on Three Datasets")
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('performance.svg', format='svg')
plt.show()