import math
import pandas as pd
from lightgbm import LGBMClassifier


def construct_model_with_decision_tree(df):
    X, y = get_x_y_columns(df)
    dt = LGBMClassifier()
    dt.fit(X, y.values.ravel())
    return dt


def get_x_y_columns(df):
    drop_columns = ['DRNumber', 'CrimeCode', 'CrimeCodeDescription', 'PremiseDescription', 'Latitude', 'Longitude']
    y_column = ['CrimeCode']

    X = df.drop(drop_columns, axis=1)
    y = df[y_column]

    return X, y


def predict_value(dt, test_data):
    X, y = get_x_y_columns(test_data)
    forecast_labels = list(dt.predict(X))
    return forecast_labels, y


def get_experiments(df):
    df = df.sample(frac=1)
    data_size = len(df)
    groups = [df.iloc[:math.ceil(data_size / 3)], df.iloc[math.ceil(data_size / 3):math.ceil((2 * data_size) / 3)],
              df.iloc[math.ceil((2 * data_size) / 3):]]
    experiments = [get_train_test_data(groups[0], groups[1], groups[2]),
                   get_train_test_data(groups[0], groups[2], groups[1]),
                   get_train_test_data(groups[1], groups[2], groups[0])]

    return experiments


def get_train_test_data(train1, train2, test):
    train_data = pd.concat([train1, train2])
    train_data = train_data.reset_index(drop=True)
    test_data = test
    test_data = test_data.reset_index(drop=True)
    return train_data, test_data


def calculate_error(y_tuple):
    return calculate_true_label_ratio(y_tuple[1], y_tuple[0])


def print_error(exp1_error, exp2_error, exp3_error):
    print('Ratio: ' + str(100 * (exp1_error + exp2_error + exp3_error) / 3) + '%')


# Accuracy
def calculate_true_label_ratio(y, forecast):
    y_size = len(y)
    return sum((1 if a == b else 0) for a, b in zip(y['CrimeCode'], forecast)) / y_size


def obtain_ohe_df(df, column_names):
    for column_name in column_names:
        # Get one hot encoding
        one_hot = pd.get_dummies(df[column_name], prefix=column_name)
        # Drop column B as it is now encoded
        df = df.drop(column_name, axis=1)
        # Join the encoded df
        df = df.join(one_hot)

    return df
