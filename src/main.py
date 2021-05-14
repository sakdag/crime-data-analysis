import understanding_dataset as ud


if __name__ == "__main__":
    # df = ud.read_dataset('crime-data-from-2010-to-present.csv')

    original_dataset_file_name = 'Crime_Data_2010_2017.csv'
    preprocessed_dataset_file_name = 'crime_data_2010_2017_preprocessed.csv'

    # ud.preprocess_and_save(original_dataset_file_name, preprocessed_dataset_file_name)

    df = ud.read_dataset(preprocessed_dataset_file_name)
    df = df.rename(columns={'DR Number': 'DRNumber', 'Time Occurred': 'TimeOccurred', 'Crime Code': 'CrimeCode',
                            'Crime Code Description': 'CrimeCodeDescription', 'Victim Age': 'VictimAge',
                            'Victim Sex': 'VictimSex', 'Victim Descent': 'VictimDescent',
                            'Premise Code': 'PremiseCode', 'Premise Description': 'PremiseDescription',
                            'Month Occurred': 'MonthOccurred', 'Day of Week': 'DayOfWeek'})

    ud.fill_with_mod(df, 'VictimDescent')
    ud.fill_with_mod(df, 'VictimSex')
    ud.fill_with_average(df, 'VictimAge', True)
