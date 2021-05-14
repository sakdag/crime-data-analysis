import understanding_dataset as ud


if __name__ == "__main__":
    # df = ud.read_dataset('crime-data-from-2010-to-present.csv')

    original_dataset_file_name = 'Crime_Data_2010_2017.csv'
    preprocessed_dataset_file_name = 'crime_data_2010_2017_preprocessed.csv'
    ud.preprocess_and_save(original_dataset_file_name, preprocessed_dataset_file_name)

    df = ud.read_dataset("crime_data_2010_2017_filtered.csv")
