import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

import src.config.config as conf
import src.preprocessing.crime_preprocessor as crime_prep
import src.utils.classification_reporter as reporter
import src.utils.visualization as visualizer


def classify_and_report(df: pd.DataFrame, number_of_folds: int, mode: str):
    columns_to_drop = {'DRNumber', 'CrimeCodeDescription',
                       'PremiseDescription', 'Latitude', 'Longitude'}
    df = df.drop(columns=columns_to_drop)

    df = crime_prep.categorize_victim_age(df)

    if mode == conf.LABEL_ENCODING:
        df[['VictimSex', 'VictimDescent', 'VictimAge']] = df[['VictimSex', 'VictimDescent', 'VictimAge']].apply(
            LabelEncoder().fit_transform)
    elif mode == conf.ONE_HOT_ENCODING:
        df = obtain_ohe_df(df, ['TimeOccurred', 'VictimAgeStage', 'VictimSex', 'VictimDescent', 'SeasonOccurred',
                                'DayOfWeek'])

    label = 'CrimeCode'
    df = df.sample(frac=1).reset_index(drop=True)
    split_dfs = np.array_split(df, number_of_folds)

    clf = LGBMClassifier()
    actual_y = list()
    predicted_y = list()

    for i in range(number_of_folds):
        # Get training set by appending elements other than current fold
        train_set = pd.DataFrame()
        for j in range(number_of_folds):
            if j != i:
                train_set = train_set.append(split_dfs[j])

        test_set = split_dfs[i]

        y = train_set[[label]]
        x = train_set.drop([label], axis=1)

        clf.fit(x, y.values.ravel())

        actual_y.extend(test_set[label].to_list())
        test_x = test_set.drop([label], axis=1)

        predicted_y.extend(list(clf.predict(test_x)))

    reporter.report(df, actual_y, predicted_y)

    confusion_matrix_answer = input("Do you want to generate the confusion matrix? (yes/no): ")

    if confusion_matrix_answer == 'yes':
        visualizer.plot_confusion_matrix(df, actual_y, predicted_y)


def obtain_ohe_df(df, column_names):
    for column_name in column_names:
        # Get one hot encoding
        one_hot = pd.get_dummies(df[column_name], prefix=column_name)
        # Drop column B as it is now encoded
        df = df.drop(column_name, axis=1)
        # Join the encoded df
        df = df.join(one_hot)

    return df
