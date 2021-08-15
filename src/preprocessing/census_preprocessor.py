import numpy as np
import pandas as pd

import src.utils.crime_classification_utils as utils
import src.config.column_names as col_names


def preprocess_and_save(original_file_name: str,
                        preprocessed_file_name: str,
                        zip_codes_dataset_file_path: str):

    census_df = utils.read_dataset(original_file_name)
    zip_codes_df = utils.read_dataset(zip_codes_dataset_file_path)

    # Insert latitude and longitude values using zip_codes_df
    insert_geolocation_info(census_df, zip_codes_df)
    print(census_df)

    # Save
    census_df.to_csv(preprocessed_file_name, index=False)


def insert_geolocation_info(df: pd.DataFrame, zip_codes_df: pd.DataFrame):
    ca_zip_codes = zip_codes_df[zip_codes_df['state_id'] == 'CA']
    df[col_names.LATITUDE] = np.NAN
    df[col_names.LONGITUDE] = np.NaN

    for census_index, census_row in df.iterrows():
        for zip_index, zip_row in ca_zip_codes.iterrows():

            if census_row[col_names.ZIP_CODE] == zip_row['zip']:
                df.loc[census_index, col_names.LATITUDE] = zip_row['lat']
                df.loc[census_index, col_names.LONGITUDE] = zip_row['lng']
                break


def categorize_total_population(df):
    low = df.loc[df[col_names.TOTAL_POPULATION] <= 20000]
    low.loc[low[col_names.TOTAL_POPULATION] <= 20000, col_names.TOTAL_POPULATION] = 'Low'

    medium = df.loc[np.logical_and(df[col_names.TOTAL_POPULATION] > 20000,
                                   df[col_names.TOTAL_POPULATION] <= 40000)]
    medium.loc[
        np.logical_and(medium[col_names.TOTAL_POPULATION] > 20000,
                       medium[col_names.TOTAL_POPULATION] <= 40000),
        col_names.TOTAL_POPULATION] = 'Medium'

    high = df.loc[np.logical_and(df[col_names.TOTAL_POPULATION] > 40000,
                                 df[col_names.TOTAL_POPULATION] <= 60000)]
    high.loc[np.logical_and(high[col_names.TOTAL_POPULATION] > 40000,
                            high[col_names.TOTAL_POPULATION] <= 60000),
              col_names.TOTAL_POPULATION] = 'High'

    extreme = df.loc[df[col_names.TOTAL_POPULATION] > 60000]
    extreme.loc[extreme[col_names.TOTAL_POPULATION] > 60000, col_names.TOTAL_POPULATION] = 'Extreme'

    low = low.append(medium, ignore_index=True)
    low = low.append(high, ignore_index=True)
    low = low.append(extreme, ignore_index=True)

    low = low.rename({col_names.TOTAL_POPULATION: col_names.TOTAL_POPULATION_CATEGORIZED}, axis=1)

    return low


def categorize_total_males_and_females(df):
    df[col_names.FEMALE_TO_MALE_RATIO_CATEGORIZED] = np.nan
    for index, row in df.iterrows():
        total_males = row[col_names.TOTAL_MALES]
        total_females = row[col_names.TOTAL_FEMALES]

        female_to_male_ratio = float(total_females + 1) / float(total_males + 1) * 100
        df.loc[index, col_names.FEMALE_TO_MALE_RATIO_CATEGORIZED] = female_to_male_ratio

    low = df.loc[df[col_names.FEMALE_TO_MALE_RATIO_CATEGORIZED] <= 48.0]
    low.loc[low[col_names.FEMALE_TO_MALE_RATIO_CATEGORIZED] <= 48.0,
            col_names.FEMALE_TO_MALE_RATIO_CATEGORIZED] = 'Low'

    almost_equal = df.loc[np.logical_and(df[col_names.FEMALE_TO_MALE_RATIO_CATEGORIZED] > 48.0,
                                   df[col_names.FEMALE_TO_MALE_RATIO_CATEGORIZED] <= 52.0)]
    almost_equal.loc[
        np.logical_and(almost_equal[col_names.FEMALE_TO_MALE_RATIO_CATEGORIZED] > 48.0,
                       almost_equal[col_names.FEMALE_TO_MALE_RATIO_CATEGORIZED] <= 52.0),
        col_names.FEMALE_TO_MALE_RATIO_CATEGORIZED] = 'AlmostEqual'

    high = df.loc[df[col_names.FEMALE_TO_MALE_RATIO_CATEGORIZED] > 52.0]
    high.loc[high[col_names.FEMALE_TO_MALE_RATIO_CATEGORIZED] > 52.0,
             col_names.FEMALE_TO_MALE_RATIO_CATEGORIZED] = 'High'

    low = low.append(almost_equal, ignore_index=True)
    low = low.append(high, ignore_index=True)

    return low


def categorize_median_age(df):
    low = df.loc[df[col_names.MEDIAN_AGE] <= 30.0]
    low.loc[low[col_names.MEDIAN_AGE] <= 30.0, col_names.MEDIAN_AGE] = 'Low'

    low_to_medium = df.loc[np.logical_and(df[col_names.MEDIAN_AGE] > 30.0,
                                          df[col_names.MEDIAN_AGE] <= 35.0)]
    low_to_medium.loc[
        np.logical_and(low_to_medium[col_names.MEDIAN_AGE] > 30.0,
                       low_to_medium[col_names.MEDIAN_AGE] <= 35.0),
        col_names.MEDIAN_AGE] = 'LowToMedium'

    medium = df.loc[np.logical_and(df[col_names.MEDIAN_AGE] > 35.0,
                                   df[col_names.MEDIAN_AGE] <= 40.0)]
    medium.loc[
        np.logical_and(medium[col_names.MEDIAN_AGE] > 35.0,
                       medium[col_names.MEDIAN_AGE] <= 40.0),
        col_names.MEDIAN_AGE] = 'Medium'

    medium_to_high = df.loc[np.logical_and(df[col_names.MEDIAN_AGE] > 40.0,
                                           df[col_names.MEDIAN_AGE] <= 45.0)]
    medium_to_high.loc[np.logical_and(medium_to_high[col_names.MEDIAN_AGE] > 40.0,
                                      medium_to_high[col_names.MEDIAN_AGE] <= 45.0),
             col_names.MEDIAN_AGE] = 'MediumToHigh'

    high = df.loc[df[col_names.MEDIAN_AGE] > 45.0]
    high.loc[high[col_names.MEDIAN_AGE] > 45.0, col_names.MEDIAN_AGE] = 'High'

    low = low.append(low_to_medium, ignore_index=True)
    low = low.append(medium, ignore_index=True)
    low = low.append(medium_to_high, ignore_index=True)
    low = low.append(high, ignore_index=True)

    low = low.rename({col_names.MEDIAN_AGE: col_names.MEDIAN_AGE_CATEGORIZED}, axis=1)

    return low
