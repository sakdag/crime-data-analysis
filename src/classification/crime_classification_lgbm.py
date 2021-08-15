import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

import src.config.config as conf
import src.preprocessing.crime_preprocessor as crime_prep
import src.utils.classification_reporter as reporter
import src.config.column_names as col_names


def classify_and_report(df: pd.DataFrame, number_of_folds: int,
                        mode: str, number_of_labels: str = conf.USE_76_LABELS, use_census: bool = False,
                        undersample: bool = True):

    if number_of_labels == conf.USE_11_LABELS:
        df = df.groupby(col_names.CRIME_CODE).filter(lambda x: len(x) > 50000)
    elif number_of_labels == conf.USE_4_LABELS:
        df = crime_prep.merge_crime_codes(df)

    if undersample:
        df = df.sample(frac=1).reset_index(drop=True)
        dfs = dict(tuple(df.groupby(col_names.CRIME_CODE)))

        # Get number of instances in the smallest df
        sample_size = min([len(current_df) for current_df in list(dfs.values())])

        # Modify each dataframe such that all have sample_size samples
        modified_dfs = [modified_df.sample(n=sample_size) for modified_df in dfs.values()]

        df = pd.concat(modified_dfs)

    columns_to_drop = {col_names.DR_NUMBER, col_names.CRIME_CODE_DESCRIPTION,
                       col_names.PREMISE_DESCRIPTION}
    df = df.drop(columns=columns_to_drop)

    if mode == conf.LABEL_ENCODING:
        df[[col_names.VICTIM_SEX, col_names.VICTIM_DESCENT, col_names.PREMISE_CODE]] = \
            df[[col_names.VICTIM_SEX,
                col_names.VICTIM_DESCENT,
                col_names.PREMISE_CODE]].apply(LabelEncoder().fit_transform)
    elif mode == conf.ONE_HOT_ENCODING:
        df = crime_prep.preprocess_and_save_before_ohe(df)
        df = obtain_ohe_df(df, ['TimeOccurred', 'VictimSex', 'VictimDescent', 'MonthOccurred', 'DayOfWeek'])

    if not use_census:
        df.drop(columns=[
            col_names.TOTAL_POPULATION,
            col_names.TOTAL_MALES,
            col_names.TOTAL_FEMALES,
            col_names.MEDIAN_AGE
        ], inplace=True)

    df.drop(columns=[
        col_names.ZIP_CODE
    ], inplace=True)

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


def obtain_ohe_df(df, column_names):
    for column_name in column_names:
        # Get one hot encoding
        one_hot = pd.get_dummies(df[column_name], prefix=column_name)
        # Drop column B as it is now encoded
        df = df.drop(column_name, axis=1)
        # Join the encoded df
        df = df.join(one_hot)

    return df
