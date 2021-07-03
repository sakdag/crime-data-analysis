import os
import sys

import src.config.config as conf
import src.preprocessing.crime_preprocessor as crime_prep
import src.preprocessing.census_preprocessor as census_prep
import src.utils.crime_classification_utils as utils
import src.correlation.dataset_correlation as correlation
import src.classification.crime_classification_categorical_nb as categorical_nb
import src.utils.visualization as visualizer

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    mode = str(sys.argv[1])

    # Original dataset file paths
    crime_dataset_file_path = os.path.join(dirname, conf.CRIME_DATASET_FILE_PATH)
    census_dataset_file_path = os.path.join(dirname, conf.CENSUS_DATASET_FILE_PATH)

    # Additional dataset file paths
    zip_codes_dataset_file_path = os.path.join(dirname, conf.ZIP_CODES_DATASET_FILE_PATH)

    # Preprocessed crime dataset file paths
    preprocessed_crime_dataset_file_path = os.path.join(dirname, conf.PREPROCESSED_CRIME_DATASET_FILE_PATH)
    preprocessed_crime_dataset_file_path_wo_nan = \
        os.path.join(dirname, conf.PREPROCESSED_CRIME_DATASET_FILE_PATH_WO_NAN)
    preprocessed_crime_dataset_file_path_ohe = \
        os.path.join(dirname, conf.PREPROCESSED_CRIME_DATASET_FILE_PATH_OHE)
    correlated_crime_dataset_file_path = os.path.join(dirname, conf.CORRELATED_CRIME_DATASET_FILE_PATH)

    # Preprocessed census dataset file paths
    preprocessed_census_dataset_file_path = os.path.join(dirname, conf.PREPROCESSED_CENSUS_DATASET_FILE_PATH)

    if mode == conf.PREPROCESS_CRIME_DATASET_MODE:
        interpolate = False
        min_number_of_rows_in_each_label = 500

        for i in range(2, len(sys.argv)):
            if sys.argv[i].split('=')[0] == 'interpolate':
                interpolate = (sys.argv[i].split('=')[1] == 'true')
            elif sys.argv[i].split('=')[0] == 'min_rows':
                min_number_of_rows_in_each_label = int(sys.argv[i].split('=')[1])

        crime_prep.preprocess_and_save(crime_dataset_file_path,
                                       preprocessed_crime_dataset_file_path,
                                       interpolate,
                                       min_number_of_rows_in_each_label)

    elif mode == conf.PREPROCESS_CENSUS_DATASET_MODE:
        census_prep.preprocess_and_save(census_dataset_file_path,
                                        preprocessed_census_dataset_file_path,
                                        zip_codes_dataset_file_path)

    elif mode == conf.CORRELATE_CRIME_DATASET_MODE:
        crime_df = utils.read_dataset(preprocessed_crime_dataset_file_path)
        census_df = utils.read_dataset(preprocessed_census_dataset_file_path)

        correlation_mode = 'haversine'

        for i in range(2, len(sys.argv)):
            if sys.argv[i].split('=')[0] == 'correlation_mode':
                correlation_mode = sys.argv[i].split('=')[1]

        correlation.correlate_and_save(crime_df,
                                       census_df,
                                       correlated_crime_dataset_file_path,
                                       correlation_mode)

    elif mode == conf.CLASSIFY_WITH_CATEGORICAL_NB_MODE:
        crime_df = utils.read_dataset(preprocessed_crime_dataset_file_path)
        number_of_folds = 3

        for i in range(2, len(sys.argv)):
            if sys.argv[i].split('=')[0] == 'number_of_folds':
                number_of_folds = int(sys.argv[i].split('=')[1])

        categorical_nb.classify_and_report(crime_df, number_of_folds)

    elif mode == conf.CLASSIFY_WITH_LGBM_MODE:
        pass

    elif mode == conf.VISUALIZE_MODE:
        crime_df = utils.read_dataset(preprocessed_crime_dataset_file_path)

        visualizer.construct_line_chart(crime_df, 'hey', 'hey', 'hey', 'hey')
        visualizer.construct_pie_chart(crime_df, 'hey')

    ####################################################################################################################
    # for the solution with one hot encoding, please read the dataset named preprocessed_crime_dataset_file_name_ohe
    # column_names = ['TimeOccurred', 'VictimAgeStage', 'VictimSex', 'VictimDescent', 'SeasonOccurred', 'DayOfWeek']
    # df = cc.obtain_ohe_df(df, column_names)

    # # for the solution with label encoder, please read the dataset named preprocessed_crime_dataset_file_name
    # df[['VictimSex', 'VictimDescent']] = df[['VictimSex', 'VictimDescent']].apply(LabelEncoder().fit_transform)
    #
    # # common part for both label encoding and one hot encoding
    # experiments = cc.get_experiments(df)
    #
    # actual_y = list()
    # predicted_y = list()
    # predicted_y_temp, actual_y_temp = \
    #     cc.predict_value(cc.construct_model_with_decision_tree(experiments[0][0]), experiments[0][1])
    # actual_y.extend(actual_y_temp['CrimeCode'].to_list())
    # predicted_y.extend(predicted_y_temp)
    # predicted_y_temp, actual_y_temp = \
    #     cc.predict_value(cc.construct_model_with_decision_tree(experiments[1][0]), experiments[1][1])
    # actual_y.extend(actual_y_temp['CrimeCode'].to_list())
    # predicted_y.extend(predicted_y_temp)
    # predicted_y_temp, actual_y_temp = \
    #     cc.predict_value(cc.construct_model_with_decision_tree(experiments[2][0]), experiments[2][1])
    # actual_y.extend(actual_y_temp['CrimeCode'].to_list())
    # predicted_y.extend(predicted_y_temp)
    # cr.report(df, actual_y, predicted_y)

    # cc.print_error(cc.calculate_error(cc.predict_value(cc.construct_model_with_decision_tree(experiments[0][0]), experiments[0][1])),
    #                cc.calculate_error(cc.predict_value(cc.construct_model_with_decision_tree(experiments[1][0]), experiments[1][1])),
    #                cc.calculate_error(cc.predict_value(cc.construct_model_with_decision_tree(experiments[2][0]), experiments[2][1])))
    ####################################################################################################################

    # What to do with these
    # ccu.print_unique_category_counts(df, 'CrimeCode')
