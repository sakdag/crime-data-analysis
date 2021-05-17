import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


def construct_pie_chart(df: pd.DataFrame, column_name: str):
    values = df[column_name].value_counts().keys().tolist()
    counts = df[column_name].value_counts().tolist()

    if len(values) > 12:
        pruned_values = values[:12]
        pruned_values.append('Others')
        values = pruned_values
        pruned_counts = counts[:12]
        pruned_counts.append(np.array(counts[12:]).sum())
        counts = pruned_counts

    counts = np.array(counts)
    patches, texts = plt.pie(counts)
    patches, labels, dummy = zip(*sorted(zip(patches, values, counts),
                                         key=lambda x: x[2],
                                         reverse=True))

    plt.legend(patches, labels, bbox_to_anchor=(-0.1, 1.), fontsize=8)

    plt.title(column_name)
    plt.show()


def plot_confusion_matrix(df: pd.DataFrame, actual_y: list, predicted_y: list):
    labels = df['CrimeCode'].unique().tolist()
    cm = confusion_matrix(actual_y, predicted_y)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()
