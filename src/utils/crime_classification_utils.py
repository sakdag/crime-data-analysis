import pandas as pd


def print_unique_category_counts(df: pd.DataFrame, column_name: str):
    print('Number of unique categories in column: ', column_name,
          ' is: ', len(df[column_name].unique()))
    print('Elements in each unique class:')
    print(df[column_name].value_counts())


def read_dataset(path):
    df = pd.read_csv(path)
    df_size = len(df)
    print('Data Size: ' + str(df_size))
    return df
