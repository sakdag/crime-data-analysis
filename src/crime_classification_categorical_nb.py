import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder


def classify_and_report(df: pd.DataFrame, number_of_folds: int):
    columns_to_drop = {'DRNumber', 'CrimeCodeDescription', 'VictimAge',
                       'PremiseDescription', 'Latitude', 'Longitude'}
    df = df.drop(columns=columns_to_drop)

    df[['VictimSex', 'VictimDescent']] = df[['VictimSex', 'VictimDescent']].apply(LabelEncoder().fit_transform)

    label = 'CrimeCode'
    split_dfs = np.array_split(df, number_of_folds)

    clf = CategoricalNB()

    number_of_right_guesses = 0

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

        test_y = test_set[label].to_list()
        test_x = test_set.drop([label], axis=1)

        predicted_y = clf.predict(test_x)

        for j in range(len(predicted_y)):
            if predicted_y[j] == test_y[j]:
                number_of_right_guesses += 1

    total_mse = float(number_of_right_guesses) / float(len(df))
    print('Total MSE: ', total_mse * 100, '%')

