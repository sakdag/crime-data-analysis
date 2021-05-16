import math
import pandas as pd
import numpy as np
import crime_classification_utils as ccu
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# Below preprocessing steps are done once, then the filtered dataset is saved to use later
def preprocess_and_save(original_file_name: str, preprocessed_file_name: str, interpolate: bool = True,
                        min_number_of_rows_in_each_label: int = 500):
    df = ccu.read_dataset(original_file_name)

    column_list = {'Date Reported', 'Area ID', 'Area Name',
                   'Reporting District', 'MO Codes', 'Weapon Used Code',
                   'Weapon Description', 'Status Code', 'Status Description',
                   'Crime Code 1', 'Crime Code 2', 'Crime Code 3', 'Crime Code 4',
                   'Address', 'Cross Street'}
    df = df.drop(columns=column_list)

    categorize_time_occurred(df)
    print(df['Time Occurred'].unique())

    create_month_occurred_column(df)
    print(df['Month Occurred'].unique())

    create_day_of_week_column(df)
    print(df['Day of Week'].unique())

    create_geolocation_columns(df)
    print(df['Longitude'])
    print(df['Latitude'])

    df = df.drop(columns={'Location ', 'Date Occurred'})

    # Remove Vehicle Stolen category of crimes as it has no victim information
    df = df[df['Crime Code'] != 510]

    # Remove rows that has unidentified crime code such as Other Miscellaneous Crime or Other Assault
    df = df[df['Crime Code'] != 946]
    df = df[df['Crime Code'] != 625]

    # Remove rows that has no location values
    df = df[df['Latitude'] != 0]
    df = df[df['Longitude'] != 0]

    df.loc[df['Victim Sex'] == 'X', 'Victim Sex'] = np.nan
    df.loc[df['Victim Descent'] == 'X', 'Victim Descent'] = np.nan

    # Temporary save
    df.to_csv(preprocessed_file_name, index=False)

    df = ccu.read_dataset(preprocessed_file_name)

    # Rename the columns
    df = df.rename(columns={'DR Number': 'DRNumber', 'Time Occurred': 'TimeOccurred', 'Crime Code': 'CrimeCode',
                            'Crime Code Description': 'CrimeCodeDescription', 'Victim Age': 'VictimAge',
                            'Victim Sex': 'VictimSex', 'Victim Descent': 'VictimDescent',
                            'Premise Code': 'PremiseCode', 'Premise Description': 'PremiseDescription',
                            'Month Occurred': 'MonthOccurred', 'Day of Week': 'DayOfWeek'})

    if interpolate:
        # Interpolate the empty values
        fill_with_mod(df, 'VictimDescent')
        fill_with_mod(df, 'VictimSex')
        fill_with_mod(df, 'PremiseCode')
        fill_with_average(df, 'VictimAge', True)
    else:
        df.dropna(subset=["VictimDescent"], inplace=True)
        df.dropna(subset=["VictimSex"], inplace=True)
        df.dropna(subset=["PremiseCode"], inplace=True)
        df.dropna(subset=["VictimAge"], inplace=True)

    # Remove crime codes that has little examples in dataset
    df = df.groupby('CrimeCode').filter(lambda x: len(x) > min_number_of_rows_in_each_label)
    df = df.groupby('VictimDescent').filter(lambda x: len(x) > 100)
    df = df.groupby('VictimSex').filter(lambda x: len(x) > 100)
    df = df.groupby('PremiseCode').filter(lambda x: len(x) > 100)

    # Save
    df.to_csv(preprocessed_file_name, index=False)


def preprocess_and_save_before_ohe(original_file_name: str, preprocessed_file_name: str):
    df = ccu.read_dataset(original_file_name)

    df = categorize_victim_age(df)

    night = df.loc[df['TimeOccurred'] <= 1]
    night.loc[night['TimeOccurred'] <= 1, 'TimeOccurred'] = 'Night'

    morning = df.loc[np.logical_and(df['TimeOccurred'] > 1, df['TimeOccurred'] <= 3)]
    morning.loc[np.logical_and(morning['TimeOccurred'] > 1, morning['TimeOccurred'] <= 3), 'TimeOccurred'] = 'Morning'

    afternoon = df.loc[np.logical_and(df['TimeOccurred'] > 3, df['TimeOccurred'] <= 5)]
    afternoon.loc[np.logical_and(afternoon['TimeOccurred'] > 3, afternoon['TimeOccurred'] <= 5), 'TimeOccurred'] = 'Afternoon'

    evening = df.loc[np.logical_and(df['TimeOccurred'] > 5, df['TimeOccurred'] <= 7)]
    evening.loc[np.logical_and(evening['TimeOccurred'] > 5, evening['TimeOccurred'] <= 7), 'TimeOccurred'] = 'Evening'

    night = night.append(morning, ignore_index=True)
    night = night.append(afternoon, ignore_index=True)
    night = night.append(evening, ignore_index=True)

    df = night

    winter = df.loc[np.logical_or(df['MonthOccurred'] <= 2, df['MonthOccurred'] == 12)]
    winter.loc[np.logical_or(winter['MonthOccurred'] <= 2, winter['MonthOccurred'] == 12), 'MonthOccurred'] = 'Winter'

    spring = df.loc[np.logical_and(df['MonthOccurred'] > 2, df['MonthOccurred'] <= 5)]
    spring.loc[np.logical_and(spring['MonthOccurred'] > 2, spring['MonthOccurred'] <= 5), 'MonthOccurred'] = 'Spring'

    summer = df.loc[np.logical_and(df['MonthOccurred'] > 5, df['MonthOccurred'] <= 8)]
    summer.loc[np.logical_and(summer['MonthOccurred'] > 5, summer['MonthOccurred'] <= 8), 'MonthOccurred'] = 'Summer'

    fall = df.loc[np.logical_and(df['MonthOccurred'] > 8, df['MonthOccurred'] <= 11)]
    fall.loc[np.logical_and(fall['MonthOccurred'] > 8, fall['MonthOccurred'] <= 11), 'MonthOccurred'] = 'Fall'

    winter = winter.append(spring, ignore_index=True)
    winter = winter.append(summer, ignore_index=True)
    winter = winter.append(fall, ignore_index=True)

    df = winter

    df = df.rename(columns={'VictimAge': 'VictimAgeStage', 'MonthOccurred': 'SeasonOccurred'})

    # Save
    df.to_csv(preprocessed_file_name, index=False)


def categorize_victim_age(df: pd.DataFrame):
    childhood = df.loc[df['VictimAge'] <= 14]
    childhood.loc[childhood['VictimAge'] <= 14, 'VictimAge'] = 'Childhood'

    adolescence = df.loc[np.logical_and(df['VictimAge'] > 14, df['VictimAge'] <= 21)]
    adolescence.loc[
        np.logical_and(adolescence['VictimAge'] > 14, adolescence['VictimAge'] <= 21), 'VictimAge'] = 'Adolescence'

    youth = df.loc[np.logical_and(df['VictimAge'] > 21, df['VictimAge'] <= 35)]
    youth.loc[np.logical_and(youth['VictimAge'] > 21, youth['VictimAge'] <= 35), 'VictimAge'] = 'Youth'

    maturity = df.loc[np.logical_and(df['VictimAge'] > 35, df['VictimAge'] <= 49)]
    maturity.loc[np.logical_and(maturity['VictimAge'] > 35, maturity['VictimAge'] <= 49), 'VictimAge'] = 'Maturity'

    aging = df.loc[np.logical_and(df['VictimAge'] > 49, df['VictimAge'] <= 63)]
    aging.loc[np.logical_and(aging['VictimAge'] > 49, aging['VictimAge'] <= 63), 'VictimAge'] = 'Aging'

    old_age = df.loc[df['VictimAge'] > 63]
    old_age.loc[old_age['VictimAge'] > 63, 'VictimAge'] = 'Old Age'

    childhood = childhood.append(adolescence, ignore_index=True)
    childhood = childhood.append(youth, ignore_index=True)
    childhood = childhood.append(maturity, ignore_index=True)
    childhood = childhood.append(aging, ignore_index=True)
    childhood = childhood.append(old_age, ignore_index=True)

    return childhood

# Print number of examples in each crime code
def print_number_of_examples_in_crime_codes(df: pd.DataFrame):
    code_dict = dict()
    for element in df['Crime Code']:
        if element in code_dict.keys():
            code_dict[element] = code_dict[element] + 1
        else:
            code_dict[element] = 1

    sorted_keys = sorted(code_dict.keys())

    for key in sorted_keys:
        print(key, "-", code_dict[key])


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


def fill_with_mod(df, column_name):
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


def get_nan_valid_tuple(df, column_name):
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


# Split crimes by their times to 8 categories (3-hour frames)
def categorize_time_occurred(df: pd.DataFrame):
    df['Time Occurred'] = df['Time Occurred'].floordiv(300)
    df['Time Occurred'] = pd.to_numeric(df['Time Occurred'])


# Create new column for month of the crime as categorical data from 0:Jan to 11:Dec
def create_month_occurred_column(df: pd.DataFrame):
    df['Date Occurred'] = pd.to_datetime(df['Date Occurred'], format='%m/%d/%Y')
    df['Month Occurred'] = df['Date Occurred'].dt.month
    df['Month Occurred'] = pd.to_numeric(df['Month Occurred'])
    df[['Month Occurred']] = df[['Month Occurred']].apply(LabelEncoder().fit_transform)


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


def impute_victim_age_using_crime_codes(df: pd.DataFrame):
    df_new = df[['Crime Code', 'Victim Age']]

    label_encoder = LabelEncoder()
    df_new['Crime Code'] = label_encoder.fit_transform(df_new['Crime Code'])
    enc = OneHotEncoder(handle_unknown='ignore')

    enc_df = pd.DataFrame(enc.fit_transform(df_new[['Crime Code']]).toarray())
    df_new = df_new.join(enc_df)
    df_new.drop(columns={'Crime Code'})
    print(df_new)

    imputer = KNNImputer(n_neighbors=3)
    df_new = imputer.fit_transform(df_new)
    print(df_new['Victim Age'].unique())
    print(df_new)
