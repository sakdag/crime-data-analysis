import math

import pandas as pd


def read_dataset(path):
    df = pd.read_csv(path)
    df_size = len(df)
    print("Data Size: " + str(df_size))
    return df


def fill_with_average(df, column_name, is_round):
    df_size = len(df)
    target_column = df[column_name]
    nan_list, value_list = get_nan_valid_tuple(df, column_name)

    avg = math.fsum(value_list) / (df_size - len(nan_list))
    if is_round:
        avg = math.ceil(avg)

    for i in nan_list:
        target_column[i] = avg


def fill_with_median(df, column_name, is_round):
    target_column = df[column_name]
    nan_list, value_list = get_nan_valid_tuple(df, column_name)

    value_list.sort()
    size = len(value_list)

    if size % 2 == 0:
        median = (value_list[int(size / 2) - 1] + value_list[int(size / 2)]) / 2
        if is_round:
            median = math.ceil(median)
    else:
        median = value_list[math.floor(size / 2)]

    for i in nan_list:
        target_column[i] = median


def get_nan_valid_tuple(df, column_name):
    df_size = len(df)
    target_column = df[column_name]
    nan_list = []
    value_list = []

    for i in range(0, df_size):
        value = target_column[i]
        if math.isnan(value):
            nan_list.append(i)
        else:
            value_list.append(i)
    return nan_list, value_list


def fill_with_median_2(df, column_name):
    target_column = df[column_name]
    indexes = pd.isnull(df).any(1).nonzero()[0]
    median = pd.DataFrame.median(df)
    for i in indexes:
        target_column[i] = median


def fill_with_mean_2(df, column_name):
    target_column = df[column_name]
    indexes = pd.isnull(df).any(1).nonzero()[0]
    mean = pd.DataFrame.mean(df)
    for i in indexes:
        target_column[i] = mean


def fill_with_knn(df, column_name, is_round):
    print("Not implemented yet")


def categorize_time_occurred(df: pd.DataFrame):
    df['Time Occurred'] = df['Time Occurred'].floordiv(300)


def create_geolocation_columns(df: pd.DataFrame):
    print("Not implemented yet")
