import numpy as np
import pandas as pd
from haversine import haversine


def correlate_and_save(df: pd.DataFrame, census_df: pd.DataFrame, file_name: str):
    # Using geolocation information in census dataset, add zipcode column to the crime dataset
    add_zip_code_column(df, census_df)
    print(df)

    # Save
    df.to_csv(file_name, index=False)


# Finds nearest zipcode geolocation to the crime location, then adds this zipcode to the crime dataset
def add_zip_code_column(df: pd.DataFrame, census_df: pd.DataFrame):
    df['Zip Code'] = np.NAN
    for index, row in df.iterrows():
        nearest_zip_code = np.NAN
        nearest_zip_code_distance = -1
        for census_index, census_row in census_df.iterrows():
            distance = haversine((row['Latitude'], row['Longitude']),
                                 (census_row['Latitude'], census_row['Longitude']))
            if nearest_zip_code_distance == -1 or distance < nearest_zip_code_distance:
                nearest_zip_code = census_row['Zip Code']
                nearest_zip_code_distance = distance
        df.loc[index, 'Zip Code'] = nearest_zip_code
