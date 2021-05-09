import understanding_dataset as ud


if __name__ == "__main__":
    # df = ud.read_dataset("crime-data-from-2010-to-present.csv")

    # # Below preprocessing steps are done once, then the filtered dataset is saved to use later
    # df = ud.read_dataset("Crime_Data_2010_2017.csv")
    #
    # column_list = {"Date Reported", "Area ID", "Area Name",
    #                "Reporting District", "MO Codes", "Weapon Used Code",
    #                "Weapon Description", "Status Code", "Status Description",
    #                "Crime Code 1", "Crime Code 2", "Crime Code 3", "Crime Code 4",
    #                "Address", "Cross Street"}
    # df = df.drop(columns=column_list)
    #
    # ud.categorize_time_occurred(df)
    # print(df['Time Occurred'].unique())
    #
    # ud.create_month_occurred_column(df)
    # print(df['Month Occurred'].unique())
    #
    # ud.create_day_of_week_column(df)
    # print(df['Day of Week'].unique())
    #
    # ud.create_geolocation_columns(df)
    # print(df['Longitude'])
    # print(df['Latitude'])
    #
    # df = df.drop(columns={'Location ', 'Date Occurred'})

    df = ud.read_dataset("crime_data_2010_2017_filtered.csv")

    # # Remove Vehicle Stolen category of crimes as it has no victim information
    # df = df[df['Crime Code'] != 510]
    #
    # # Remove rows that has unidentified crime code such as Other Miscellaneous Crime or Other Assault
    # df = df[df['Crime Code'] != 946]
    # df = df[df['Crime Code'] != 625]
    #
    # # Remove rows that has no location values
    # df = df[df['Latitude'] != 0]
    # df = df[df['Longitude'] != 0]
    #
    # # Remove crime codes that has little examples in dataset
    # df = df.groupby('Crime Code').filter(lambda x: len(x) > 500)
    #
    # df.to_csv(r'crime_data_2010_2017_filtered.csv', index=False)
    #
    # # Print number of examples in each crime code
    # code_dict = dict()
    # for element in df['Crime Code']:
    #     if element in code_dict.keys():
    #         code_dict[element] = code_dict[element] + 1
    #     else:
    #         code_dict[element] = 1
    #
    # sorted_keys = sorted(code_dict.keys())
    #
    # for key in sorted_keys:
    #     print(key, "-", code_dict[key])

    ud.impute_victim_age_using_crime_codes(df)

    # ud.fill_with_average(df, 'Victim Age', True)
