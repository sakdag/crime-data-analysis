import understanding_dataset as ud


if __name__ == "__main__":
    # df = ud.read_dataset("crime-data-from-2010-to-present.csv")
    # df = ud.read_dataset("Crime_Data_2010_2017.csv")
    #
    # column_list = {"Date Reported", "Date Occurred", "Area ID", "Area Name",
    #                "Reporting District", "MO Codes", "Weapon Used Code",
    #                "Weapon Description", "Status Code", "Status Description",
    #                "Crime Code 1", "Crime Code 2", "Crime Code 3", "Crime Code 4",
    #                "Address", "Cross Street"}
    # df = df.drop(columns=column_list)
    #
    # filtered_csv = df.to_csv(r'crime_data_2010_2017_filtered.csv', index=False)

    df = ud.read_dataset("crime_data_2010_2017_filtered.csv")

    ud.categorize_time_occurred(df)
    print(df['Time Occurred'].unique())

    # ud.fill_with_average(df, 'Victim Age', True)
