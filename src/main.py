import understanding_dataset as ud
import census_preprocessor as cp
import dataset_correlation as dc


if __name__ == "__main__":
    crime_dataset_file_name = 'Crime_Data_2010_2017.csv'
    preprocessed_crime_dataset_file_name = 'crime_data_2010_2017_preprocessed.csv'
    correlated_crime_dataset_file_name = 'crime_data_2010_2017_correlated.csv'

    census_dataset_file_name = '2010-census-populations-by-zip-code.csv'
    preprocessed_census_dataset_file_name = '2010-census-populations-by-zip-code-preprocessed.csv'

    # ud.preprocess_and_save(original_dataset_file_name, preprocessed_dataset_file_name)
    # cp.preprocess_and_save(census_dataset_file_name, preprocessed_census_dataset_file_name)

    df = ud.read_dataset(preprocessed_crime_dataset_file_name)
    census_df = ud.read_dataset(preprocessed_census_dataset_file_name)

    dc.correlate_and_save(df, census_df, correlated_crime_dataset_file_name)
