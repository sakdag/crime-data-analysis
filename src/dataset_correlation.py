import numpy as np
import pandas as pd
import pyproj as pyproj
import shapely as shapely
from shapely import ops
from shapely.geometry import Point

import crime_classification_utils as ccu
from haversine import haversine


def correlate_and_save(df: pd.DataFrame, census_df: pd.DataFrame, file_name: str):
    map_file_name = 'location_zipcode_map_no_duplicates.csv'

    # Using geolocation information in census dataset, add zipcode column to the crime dataset
    # add_zip_code_column_using_map(df, map_file_name)
    # add_zip_code_column(df, census_df)
    add_zip_code_column_by_calculating_euclidean(df, census_df)

    # Save
    df.to_csv(file_name, index=False)


# As add_zip_code_column_by_calculating takes too much time, its result is saved to map.
# This method uses that map instead of calculating again to insert zip code column to dataset
def add_zip_code_column_using_map(df: pd.DataFrame, map_file_name: str):
    map_df = ccu.read_dataset(map_file_name)
    map_df = map_df.rename(columns={'Zip Code': 'ZipCode'})

    # TODO: Not working correctly
    df['ZipCode'] = df.merge(map_df, on=['Latitude', 'Longitude'])['ZipCode']


# Finds nearest zipcode geolocation to the crime location, then adds this zipcode to the crime dataset
def add_zip_code_column_by_calculating(df: pd.DataFrame, census_df: pd.DataFrame):
    df['Zip Code'] = np.nan
    for index, row in df.iterrows():
        nearest_zip_code = np.nan
        nearest_zip_code_distance = -1
        for census_index, census_row in census_df.iterrows():
            distance = haversine((row['Latitude'], row['Longitude']),
                                 (census_row['Latitude'], census_row['Longitude']))
            if nearest_zip_code_distance == -1 or distance < nearest_zip_code_distance:
                nearest_zip_code = census_row['Zip Code']
                nearest_zip_code_distance = distance
        df.loc[index, 'Zip Code'] = nearest_zip_code


def add_zip_code_column_by_calculating_euclidean(df: pd.DataFrame, census_df: pd.DataFrame):
    df['Zip Code'] = np.nan
    wgs84_proj = pyproj.CRS('EPSG:4326')
    los_angeles_proj = pyproj.CRS('EPSG:6423')
    project_los_angeles = pyproj.Transformer.from_crs(wgs84_proj, los_angeles_proj, always_xy=True).transform
    census_list = list()
    for census_index, census_row in census_df.iterrows():
        point2_transformed = shapely.ops.transform(project_los_angeles, Point(census_row['Latitude'], census_row['Longitude']))
        census_list.append((census_row, point2_transformed))

    for index, row in df.iterrows():
        nearest_zip_code = np.nan
        nearest_zip_code_distance = -1
        point1_transformed = shapely.ops.transform(project_los_angeles, Point(row['Latitude'], row['Longitude']))
        for census_data in census_list:
            distance = point1_transformed.distance(census_data[1])
            if nearest_zip_code_distance == -1 or distance < nearest_zip_code_distance:
                nearest_zip_code = census_data[0]['Zip Code']
                nearest_zip_code_distance = distance
        df.loc[index, 'Zip Code'] = nearest_zip_code
