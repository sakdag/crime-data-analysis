import pandas as pd
from sklearn import metrics

import src.utils.visualization as visualizer


def report(df: pd.DataFrame, actual_y: list, predicted_y: list):
    # Print the precision and recall, among other metrics
    print(metrics.classification_report(actual_y, predicted_y, digits=3))

    confusion_matrix_answer = input("Do you want to generate the confusion matrix? (yes/no): ")
    if confusion_matrix_answer == 'yes':
        visualizer.plot_confusion_matrix(df, actual_y, predicted_y)
