import understanding_dataset as ud


if __name__ == "__main__":
    # df = ud.read_dataset("crime-data-from-2010-to-present.csv")

    # Below preprocessing steps are done once, then the filtered dataset is saved to use later
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
    # df.to_csv(r'crime_data_2010_2017_filtered.csv', index=False)

    df = ud.read_dataset("crime_data_2010_2017_filtered.csv")

    # ud.fill_with_average(df, 'Victim Age', True)
