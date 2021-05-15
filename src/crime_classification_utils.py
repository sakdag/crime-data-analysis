import pandas as pd


def print_unique_category_counts(df: pd.DataFrame, column_name: str):
    print('Number of unique categories in column: ', column_name,
          ' is: ', len(df[str].unique()))
    print('Elements in each unique class:')
    print(df[str].value_counts())
