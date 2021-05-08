import math
import pandas as pd


def read_dataset(path):
    df = pd.read_csv(path)
    df_size = len(df)
    print("Data Size: " + str(df_size))


def fill_with_average(df, column_name, is_round):
    df_size = len(df)
    target_column = df[column_name]
    nan_list = []
    total = 0
    for i in range(0, df_size):
        value = target_column[i]
        if math.isnan(value):
            nan_list.append(i)
        else:
            total = total + value

    avg = total / (df_size - len(nan_list))
    if is_round:
        avg = math.ceil(avg)

    for i in nan_list:
        target_column[i] = avg

    for i in range(0, df_size):
        value = df[column_name][i]
        if math.isnan(value):
            print("You copied, silly!")

