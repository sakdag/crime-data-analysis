from sklearn import metrics


def report(actual_y: list, predicted_y: list):

    # Print total accuracy
    # accuracy = metrics.accuracy_score(actual_y, predicted_y)
    # print('Total Accuracy: ', accuracy * 100, '%')

    # Print confusion metric
    # print(metrics.confusion_matrix(actual_y, predicted_y))

    # Print the precision and recall, among other metrics
    print(metrics.classification_report(actual_y, predicted_y, digits=3))
