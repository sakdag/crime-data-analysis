import math
import timeit
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def read_dataset(path):
    df = pd.read_csv(path)
    df_size = len(df)
    print("Data Size: " + str(df_size))
    return df


# is_round: if column values are not continuous, then is_round must be true
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
            value_list.append(target_column[i])
    return nan_list, value_list


'''
def fill_with_median_2(df, column_name):
    target_column = df[column_name]
    indexes = df[column_name].index[df[column_name].apply(np.isnan)]
    median = pd.DataFrame.median(target_column)
    for i in indexes:
        target_column[i] = median


def fill_with_mean_2(df, column_name):
    target_column = df[column_name]
    indexes = df[column_name].index[df[column_name].apply(np.isnan)]
    mean = pd.DataFrame.mean(target_column)
    for i in indexes:
        target_column[i] = mean
'''


# WRONG
def fill_with_knn_numeric(df, column_name):
    target_column = df[column_name]
    imputer = KNNImputer(n_neighbors=2)
    return imputer.fit_transform([target_column])


# WRONG
def fill_with_knn_categorical(df, column_name):
    mapped_column = df.cat.codes
    imputer = KNNImputer(n_neighbors=2)
    imputer.fit_transform(mapped_column)
    return mapped_column


# Split crimes by their times to 8 categories (3-hour frames)
def categorize_time_occurred(df: pd.DataFrame):
    df['Time Occurred'] = df['Time Occurred'].floordiv(300)
    df['Time Occurred'] = pd.to_numeric(df['Time Occurred'])


# Create new column for month of the crime as categorical data from 0:Jan to 11:Dec
def create_month_occurred_column(df: pd.DataFrame):
    df['Date Occurred'] = pd.to_datetime(df['Date Occurred'], format='%m/%d/%Y')
    df['Month Occurred'] = df['Date Occurred'].dt.month
    df['Month Occurred'] = pd.to_numeric(df['Month Occurred'])


# Create new column which describes which day of week the crime occurred, 0:Mon to 6:Sun
def create_day_of_week_column(df: pd.DataFrame):
    df['Day of Week'] = df['Date Occurred'].dt.dayofweek
    df['Day of Week'] = pd.to_numeric(df['Day of Week'])


# Extract Lat-Long information from Location column and create columns for them to use later
def create_geolocation_columns(df: pd.DataFrame):
    df['Location '].replace(np.NaN, '(0,0)')
    df[['Latitude', 'Longitude']] = df['Location '].str.split(',', expand=True)
    df['Latitude'] = df['Latitude'].str[1:]
    df['Longitude'] = df['Longitude'].str[:-1]
    df['Longitude'] = pd.to_numeric(df['Longitude'])
    df['Latitude'] = pd.to_numeric(df['Latitude'])
