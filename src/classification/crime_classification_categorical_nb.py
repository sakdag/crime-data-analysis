import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

import src.utils.classification_reporter as reporter
import src.preprocessing.crime_preprocessor as crime_prep
import src.preprocessing.census_preprocessor as census_prep
import src.config.column_names as col_names
import src.config.config as conf


def classify_and_report(df: pd.DataFrame, number_of_folds: int,
                        number_of_labels: str = conf.USE_72_LABELS, use_census: bool = False,
                        undersample: bool = True):

    if number_of_labels == conf.USE_11_LABELS:
        df = df.groupby(col_names.CRIME_CODE).filter(lambda x: len(x) > 50000)
    elif number_of_labels == conf.USE_5_LABELS:
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

    df = crime_prep.categorize_victim_age(df)

    df[[col_names.VICTIM_SEX, col_names.VICTIM_DESCENT, col_names.VICTIM_AGE, col_names.PREMISE_CODE]] = \
        df[[col_names.VICTIM_SEX,
            col_names.VICTIM_DESCENT,
            col_names.VICTIM_AGE,
            col_names.PREMISE_CODE]].apply(LabelEncoder().fit_transform)

    if use_census:
        df = census_prep.categorize_total_population(df)
        # df = census_prep.categorize_total_males_and_females(df)
        df = census_prep.categorize_median_age(df)

        df[[col_names.ZIP_CODE, col_names.TOTAL_POPULATION_CATEGORIZED, col_names.MEDIAN_AGE_CATEGORIZED]] = \
            df[[col_names.ZIP_CODE,
                col_names.TOTAL_POPULATION_CATEGORIZED,
                col_names.MEDIAN_AGE_CATEGORIZED]].apply(LabelEncoder().fit_transform)

        df.drop(columns=[
            col_names.ZIP_CODE,
            col_names.TOTAL_MALES,
            col_names.TOTAL_FEMALES
        ], inplace=True)

    else:
        df.drop(columns=[
            col_names.ZIP_CODE,
            col_names.TOTAL_POPULATION,
            col_names.TOTAL_MALES,
            col_names.TOTAL_FEMALES,
            col_names.MEDIAN_AGE
        ], inplace=True)

    label = col_names.CRIME_CODE
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
