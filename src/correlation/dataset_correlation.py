import numpy as np
import pandas as pd
import pyproj as pyproj
import shapely as shapely
from shapely.geometry import Point
from haversine import haversine

import src.config.column_names as col_names


def correlate_and_save(crime_df: pd.DataFrame,
                       census_df: pd.DataFrame,
                       file_name: str,
                       correlation_mode: str):

    # Using geolocation information in census dataset, add zipcode column to the crime dataset
    if correlation_mode == 'euclidean':
        add_zip_code_column_using_euclidean(crime_df, census_df)
    else:
        add_zip_code_column_using_haversine(crime_df, census_df)

    # Save
    crime_df.to_csv(file_name, index=False)


# Finds nearest zipcode geolocation to the crime location, then adds this zipcode to the crime dataset
def add_zip_code_column_using_haversine(crime_df: pd.DataFrame, census_df: pd.DataFrame):
    crime_df[col_names.ZIP_CODE] = np.nan
    for index, row in crime_df.iterrows():
        nearest_zip_code = np.nan
        nearest_zip_code_distance = -1
        for census_index, census_row in census_df.iterrows():
            distance = haversine((row[col_names.LATITUDE], row[col_names.LONGITUDE]),
                                 (census_row[col_names.LATITUDE], census_row[col_names.LONGITUDE]))
            if nearest_zip_code_distance == -1 or distance < nearest_zip_code_distance:
                nearest_zip_code = census_row[col_names.ZIP_CODE]
                nearest_zip_code_distance = distance
        crime_df.loc[index, col_names.ZIP_CODE] = nearest_zip_code


def add_zip_code_column_using_euclidean(crime_df: pd.DataFrame, census_df: pd.DataFrame):
    crime_df[col_names.ZIP_CODE] = np.nan
    wgs84_proj = pyproj.CRS('EPSG:4326')
    los_angeles_proj = pyproj.CRS('EPSG:6423')
    project_los_angeles = pyproj.Transformer.from_crs(wgs84_proj, los_angeles_proj, always_xy=True).transform
    census_list = list()
    for census_index, census_row in census_df.iterrows():
        point2_transformed = shapely.ops.transform(project_los_angeles,
                                                   Point(census_row[col_names.LATITUDE], census_row[col_names.LONGITUDE]))
        census_list.append((census_row, point2_transformed))

    for index, row in crime_df.iterrows():
        nearest_zip_code = np.nan
        nearest_zip_code_distance = -1
        point1_transformed = shapely.ops.transform(project_los_angeles, Point(row[col_names.LATITUDE], row[col_names.LONGITUDE]))
        for census_data in census_list:
            distance = point1_transformed.distance(census_data[1])
            if nearest_zip_code_distance == -1 or distance < nearest_zip_code_distance:
                nearest_zip_code = census_data[0][col_names.ZIP_CODE]
                nearest_zip_code_distance = distance
        crime_df.loc[index, col_names.ZIP_CODE] = nearest_zip_code


def merge_crime_and_census(crime_df: pd.DataFrame, census_df: pd.DataFrame, file_name: str):
    merged_df = crime_df.merge(census_df, on=col_names.ZIP_CODE, how='inner')
    redundant_columns = ['Latitude_x', 'Latitude_y', 'Longitude_x', 'Longitude_y',
                         col_names.TOTAL_HOUSEHOLDS, col_names.AVERAGE_HOUSEHOLD_SIZE]
    merged_df.drop(columns=redundant_columns, inplace=True)

    # Save
    merged_df.to_csv(file_name, index=False)
