import numpy as np
import pandas as pd
import crime_preprocessor as ud
import crime_classification_utils as ccu


def preprocess_and_save(original_file_name: str, preprocessed_file_name: str):
    census_df = ccu.read_dataset(original_file_name)
    zip_codes_dataset_file_name = 'uszips.csv'
    zip_codes_df = ccu.read_dataset(zip_codes_dataset_file_name)

    insert_geolocation_info(census_df, zip_codes_df)
    print(census_df)

    # Save
    census_df.to_csv(preprocessed_file_name, index=False)


def insert_geolocation_info(df: pd.DataFrame, zip_codes_df: pd.DataFrame):
    ca_zip_codes = zip_codes_df[zip_codes_df['state_id'] == 'CA']
    df['Latitude'] = np.NAN
    df['Longitude'] = np.NaN
    for census_index, census_row in df.iterrows():
        for zip_index, zip_row in ca_zip_codes.iterrows():
            if census_row['Zip Code'] == zip_row['zip']:
                df.loc[census_index, 'Latitude'] = zip_row['lat']
                df.loc[census_index, 'Longitude'] = zip_row['lng']
                break
