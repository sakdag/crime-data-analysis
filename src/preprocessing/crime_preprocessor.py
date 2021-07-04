import math
import pandas as pd
import numpy as np

from src.utils import crime_classification_utils as ccu
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import src.config.crime_code_constants as crime_const
import src.config.column_names as col_names


# Below preprocessing steps are done once, then the filtered dataset is saved to use later
def preprocess_and_save(original_file_name: str,
                        preprocessed_file_name: str,
                        interpolate: bool = True,
                        min_number_of_rows_in_each_label: int = 500):
    df = ccu.read_dataset(original_file_name)

    # Drop unused attribute columns
    df = df.drop(columns=col_names.UNUSED_COLUMN_LIST)

    categorize_time_occurred(df)
    print(df[col_names.TIME_OCCURRED].unique())

    create_month_occurred_column(df)
    print(df[col_names.MONTH_OCCURRED].unique())

    create_day_of_week_column(df)
    print(df[col_names.DAY_OF_WEEK].unique())

    create_geolocation_columns(df)
    print(df[col_names.LONGITUDE])
    print(df[col_names.LATITUDE])

    df = df.drop(columns={col_names.LOCATION, col_names.DATE_OCCURRED})

    # Remove Vehicle Stolen category of crimes as it has no victim information
    df = df[df[col_names.CRIME_CODE] != 510]

    # Remove rows that has unidentified crime code such as Other Miscellaneous Crime or Other Assault
    df = df[df[col_names.CRIME_CODE] != 946]
    df = df[df[col_names.CRIME_CODE] != 625]

    # Remove rows that has no location values
    df = df[df[col_names.LATITUDE] != 0]
    df = df[df[col_names.LONGITUDE] != 0]

    df.loc[df[col_names.VICTIM_SEX] == 'X', col_names.VICTIM_SEX] = np.nan
    df.loc[df[col_names.VICTIM_DESCENT] == 'X', col_names.VICTIM_DESCENT] = np.nan

    # Temporary save
    df.to_csv(preprocessed_file_name, index=False)

    df = ccu.read_dataset(preprocessed_file_name)

    if interpolate:
        # Interpolate the empty values
        fill_with_mod(df, col_names.VICTIM_DESCENT)
        fill_with_mod(df, col_names.VICTIM_SEX)
        fill_with_mod(df, col_names.PREMISE_CODE)
        fill_with_average(df, col_names.VICTIM_AGE, True)
    else:
        df.dropna(subset=[col_names.VICTIM_DESCENT], inplace=True)
        df.dropna(subset=[col_names.VICTIM_SEX], inplace=True)
        df.dropna(subset=[col_names.PREMISE_CODE], inplace=True)
        df.dropna(subset=[col_names.VICTIM_AGE], inplace=True)

    # Remove crime codes that has little examples in dataset
    df = df.groupby(col_names.CRIME_CODE).filter(lambda x: len(x) > min_number_of_rows_in_each_label)
    df = df.groupby(col_names.VICTIM_DESCENT).filter(lambda x: len(x) > 100)
    df = df.groupby(col_names.VICTIM_SEX).filter(lambda x: len(x) > 100)
    df = df.groupby(col_names.PREMISE_CODE).filter(lambda x: len(x) > 100)

    # Save
    df.to_csv(preprocessed_file_name, index=False)


def preprocess_and_save_before_ohe(df: pd.DataFrame):

    df = categorize_victim_age(df)

    night = df.loc[df[col_names.TIME_OCCURRED] <= 1]
    night.loc[night[col_names.TIME_OCCURRED] <= 1, col_names.TIME_OCCURRED] = 'Night'

    morning = df.loc[np.logical_and(df[col_names.TIME_OCCURRED] > 1, df[col_names.TIME_OCCURRED] <= 3)]
    morning.loc[np.logical_and(morning[col_names.TIME_OCCURRED] > 1, morning[col_names.TIME_OCCURRED] <= 3),
                col_names.TIME_OCCURRED] = 'Morning'

    afternoon = df.loc[np.logical_and(df[col_names.TIME_OCCURRED] > 3, df[col_names.TIME_OCCURRED] <= 5)]
    afternoon.loc[np.logical_and(afternoon[col_names.TIME_OCCURRED] > 3, afternoon[col_names.TIME_OCCURRED] <= 5),
                  col_names.TIME_OCCURRED] = 'Afternoon'

    evening = df.loc[np.logical_and(df[col_names.TIME_OCCURRED] > 5, df[col_names.TIME_OCCURRED] <= 7)]
    evening.loc[np.logical_and(evening[col_names.TIME_OCCURRED] > 5, evening[col_names.TIME_OCCURRED] <= 7),
                col_names.TIME_OCCURRED] = 'Evening'

    night = night.append(morning, ignore_index=True)
    night = night.append(afternoon, ignore_index=True)
    night = night.append(evening, ignore_index=True)

    df = night

    winter = df.loc[np.logical_or(df[col_names.MONTH_OCCURRED] <= 2, df[col_names.MONTH_OCCURRED] == 12)]
    winter.loc[np.logical_or(winter[col_names.MONTH_OCCURRED] <= 2, winter[col_names.MONTH_OCCURRED] == 12),
               col_names.MONTH_OCCURRED] = 'Winter'

    spring = df.loc[np.logical_and(df[col_names.MONTH_OCCURRED] > 2, df[col_names.MONTH_OCCURRED] <= 5)]
    spring.loc[np.logical_and(spring[col_names.MONTH_OCCURRED] > 2, spring[col_names.MONTH_OCCURRED] <= 5),
               col_names.MONTH_OCCURRED] = 'Spring'

    summer = df.loc[np.logical_and(df[col_names.MONTH_OCCURRED] > 5, df[col_names.MONTH_OCCURRED] <= 8)]
    summer.loc[np.logical_and(summer[col_names.MONTH_OCCURRED] > 5, summer[col_names.MONTH_OCCURRED] <= 8),
               col_names.MONTH_OCCURRED] = 'Summer'

    fall = df.loc[np.logical_and(df[col_names.MONTH_OCCURRED] > 8, df[col_names.MONTH_OCCURRED] <= 11)]
    fall.loc[np.logical_and(fall[col_names.MONTH_OCCURRED] > 8, fall[col_names.MONTH_OCCURRED] <= 11),
             col_names.MONTH_OCCURRED] = 'Fall'

    winter = winter.append(spring, ignore_index=True)
    winter = winter.append(summer, ignore_index=True)
    winter = winter.append(fall, ignore_index=True)

    df = winter

    return df


def categorize_victim_age(df: pd.DataFrame):
    childhood = df.loc[df[col_names.VICTIM_AGE] <= 14]
    childhood.loc[childhood[col_names.VICTIM_AGE] <= 14, col_names.VICTIM_AGE] = 'Childhood'

    adolescence = df.loc[np.logical_and(df[col_names.VICTIM_AGE] > 14, df[col_names.VICTIM_AGE] <= 21)]
    adolescence.loc[
        np.logical_and(adolescence[col_names.VICTIM_AGE] > 14, adolescence[col_names.VICTIM_AGE] <= 21),
        col_names.VICTIM_AGE] = 'Adolescence'

    youth = df.loc[np.logical_and(df[col_names.VICTIM_AGE] > 21, df[col_names.VICTIM_AGE] <= 35)]
    youth.loc[np.logical_and(youth[col_names.VICTIM_AGE] > 21, youth[col_names.VICTIM_AGE] <= 35),
              col_names.VICTIM_AGE] = 'Youth'

    maturity = df.loc[np.logical_and(df[col_names.VICTIM_AGE] > 35, df[col_names.VICTIM_AGE] <= 49)]
    maturity.loc[np.logical_and(maturity[col_names.VICTIM_AGE] > 35, maturity[col_names.VICTIM_AGE] <= 49),
                 col_names.VICTIM_AGE] = 'Maturity'

    aging = df.loc[np.logical_and(df[col_names.VICTIM_AGE] > 49, df[col_names.VICTIM_AGE] <= 63)]
    aging.loc[np.logical_and(aging[col_names.VICTIM_AGE] > 49, aging[col_names.VICTIM_AGE] <= 63),
              col_names.VICTIM_AGE] = 'Aging'

    old_age = df.loc[df[col_names.VICTIM_AGE] > 63]
    old_age.loc[old_age[col_names.VICTIM_AGE] > 63, col_names.VICTIM_AGE] = 'Old Age'

    childhood = childhood.append(adolescence, ignore_index=True)
    childhood = childhood.append(youth, ignore_index=True)
    childhood = childhood.append(maturity, ignore_index=True)
    childhood = childhood.append(aging, ignore_index=True)
    childhood = childhood.append(old_age, ignore_index=True)

    return childhood


# is_round: if column values are not continuous, then is_round must be true
def fill_with_average(df: pd.DataFrame, column_name: str, is_round: bool):
    df_size = len(df)
    target_column = df[column_name]
    nan_list, value_list = get_nan_valid_tuple(df, column_name)

    avg = math.fsum(value_list) / (df_size - len(nan_list))
    if is_round:
        avg = math.ceil(avg)

    for i in nan_list:
        target_column[i] = avg


def fill_with_median(df: pd.DataFrame, column_name: str, is_round: bool):
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


def fill_with_mod(df: pd.DataFrame, column_name: str):
    target_column = df[column_name]
    nan_list, value_list = get_nan_valid_tuple(df, column_name)

    groups = df.groupby(column_name)
    mod = ''
    mod_value = 0
    for sub in groups:
        len_sub = len(sub[1])
        if mod_value <= len_sub:
            mod_value = len_sub
            mod = sub[0]

    for i in nan_list:
        target_column[i] = mod


def get_nan_valid_tuple(df: pd.DataFrame, column_name: str):
    df_size = len(df)
    target_column = df[column_name]
    nan_list = []
    value_list = []

    for i in range(0, df_size):
        value = target_column[i]
        if isinstance(value, str):
            if value == np.nan:
                nan_list.append(i)
            else:
                value_list.append(target_column[i])
        elif math.isnan(value):
            nan_list.append(i)
        else:
            value_list.append(target_column[i])
    return nan_list, value_list


# Split crimes by their times to 8 categories (3-hour frames)
def categorize_time_occurred(df: pd.DataFrame):
    df[col_names.TIME_OCCURRED] = df[col_names.TIME_OCCURRED].floordiv(300)
    df[col_names.TIME_OCCURRED] = pd.to_numeric(df[col_names.TIME_OCCURRED])


# Create new column for month of the crime as categorical data from 0:Jan to 11:Dec
def create_month_occurred_column(df: pd.DataFrame):
    df[col_names.DATE_OCCURRED] = pd.to_datetime(df[col_names.DATE_OCCURRED], format='%m/%d/%Y')
    df[col_names.MONTH_OCCURRED] = df[col_names.DATE_OCCURRED].dt.month
    df[col_names.MONTH_OCCURRED] = pd.to_numeric(df[col_names.MONTH_OCCURRED])
    df[[col_names.MONTH_OCCURRED]] = df[[col_names.MONTH_OCCURRED]].apply(LabelEncoder().fit_transform)


# Create new column which describes which day of week the crime occurred, 0:Mon to 6:Sun
def create_day_of_week_column(df: pd.DataFrame):
    df[col_names.DAY_OF_WEEK] = df[col_names.DATE_OCCURRED].dt.dayofweek
    df[col_names.DAY_OF_WEEK] = pd.to_numeric(df[col_names.DAY_OF_WEEK])


# Extract Lat-Long information from Location column and create columns for them to use later
def create_geolocation_columns(df: pd.DataFrame):
    df[col_names.LOCATION].replace(np.NaN, '(0,0)')
    df[[col_names.LATITUDE, col_names.LONGITUDE]] = df[col_names.LOCATION].str.split(',', expand=True)
    df[col_names.LATITUDE] = df[col_names.LATITUDE].str[1:]
    df[col_names.LONGITUDE] = df[col_names.LONGITUDE].str[:-1]
    df[col_names.LONGITUDE] = pd.to_numeric(df[col_names.LONGITUDE])
    df[col_names.LATITUDE] = pd.to_numeric(df[col_names.LATITUDE])


def impute_victim_age_using_crime_codes(df: pd.DataFrame):
    df_new = df[[col_names.CRIME_CODE, col_names.VICTIM_AGE]]

    label_encoder = LabelEncoder()
    df_new[col_names.CRIME_CODE] = label_encoder.fit_transform(df_new[col_names.CRIME_CODE])
    enc = OneHotEncoder(handle_unknown='ignore')

    enc_df = pd.DataFrame(enc.fit_transform(df_new[[col_names.CRIME_CODE]]).toarray())
    df_new = df_new.join(enc_df)
    df_new.drop(columns={col_names.CRIME_CODE})
    print(df_new)

    imputer = KNNImputer(n_neighbors=3)
    df_new = imputer.fit_transform(df_new)
    print(df_new[col_names.VICTIM_AGE].unique())
    print(df_new)


def merge_crime_codes_and_save(df: pd.DataFrame, csv_name: str):
    for i in range(len(crime_const.MERGED_CRIME_DESC_CODES)):
        df.loc[
            (df[col_names.CRIME_CODE_DESCRIPTION].isin(crime_const.CRIME_LIST[i])), col_names.CRIME_CODE_DESCRIPTION] = \
            crime_const.MERGED_CRIME_DESC_CODES[i][0]
        df.loc[
            (df[col_names.CRIME_CODE_DESCRIPTION] == crime_const.MERGED_CRIME_DESC_CODES[i][0]), col_names.CRIME_CODE] = \
            crime_const.MERGED_CRIME_DESC_CODES[i][1]

    # Save
    df.to_csv(csv_name, index=False)
