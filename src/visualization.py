import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


def construct_pie_chart(df: pd.DataFrame, column_name: str):
    value_counts = df[column_name].value_counts()
    y = []
    labels = []
    for group in value_counts:
        labels.append(group[0])
        y.append(group[1])

    y = np.array(y)
    plt.pie(y, labels=labels)
    plt.show()


def plot_confusion_matrix(df: pd.DataFrame, actual_y: list, predicted_y: list):
    labels = df['CrimeCode'].unique().tolist()
    cm = confusion_matrix(actual_y, predicted_y)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()

