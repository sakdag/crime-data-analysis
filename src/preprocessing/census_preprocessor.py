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
