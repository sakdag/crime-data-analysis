import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

import src.utils.classification_reporter as reporter
import src.preprocessing.crime_preprocessor as crime_prep


def classify_and_report(df: pd.DataFrame, number_of_folds: int):
    columns_to_drop = {'DRNumber', 'CrimeCodeDescription',
                       'PremiseDescription', 'Latitude', 'Longitude'}
    df = df.drop(columns=columns_to_drop)

    df = crime_prep.categorize_victim_age(df)

    df[['VictimSex', 'VictimDescent', 'VictimAge']] = \
        df[['VictimSex', 'VictimDescent', 'VictimAge']].apply(LabelEncoder().fit_transform)

    label = 'CrimeCode'
    df = df.sample(frac=1).reset_index(drop=True)
    split_dfs = np.array_split(df, number_of_folds)

    clf = CategoricalNB()
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
