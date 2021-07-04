import os
import sys

import src.config.config as conf
import src.config.crime_code_constants as constants
import src.preprocessing.crime_preprocessor as crime_prep
import src.preprocessing.census_preprocessor as census_prep
import src.utils.crime_classification_utils as utils
import src.correlation.dataset_correlation as correlation
import src.classification.crime_classification_categorical_nb as categorical_nb
import src.classification.crime_classification_lgbm as lgbm
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

    elif mode == conf.MERGE_DATASETS_MODE:
        crime_df = utils.read_dataset(correlated_crime_dataset_file_path)
        census_df = utils.read_dataset(preprocessed_census_dataset_file_path)

        correlation.merge_crime_and_census(crime_df, census_df, preprocessed_crime_dataset_file_path)

    elif mode == conf.CLASSIFY_WITH_CATEGORICAL_NB_MODE:
        crime_df = utils.read_dataset(preprocessed_crime_dataset_file_path)
        number_of_folds = 3
        use_census = False
        number_of_labels = conf.USE_72_LABELS
        undersample = False

        for i in range(2, len(sys.argv)):
            if sys.argv[i].split('=')[0] == 'number_of_folds':
                number_of_folds = int(sys.argv[i].split('=')[1])
            if sys.argv[i].split('=')[0] == 'use_census':
                use_census = (sys.argv[i].split('=')[1] == 'true')
            if sys.argv[i].split('=')[0] == 'number_of_labels':
                number_of_labels = sys.argv[i].split('=')[1]
            if sys.argv[i].split('=')[0] == 'undersample':
                undersample = (sys.argv[i].split('=')[1] == 'true')

        categorical_nb.classify_and_report(crime_df, number_of_folds, number_of_labels, use_census, undersample)

    elif mode == conf.CLASSIFY_WITH_LGBM_MODE:
        crime_df = utils.read_dataset(preprocessed_crime_dataset_file_path)
        number_of_folds = 3
        categorical_column_handling_method = 'label_encoding'

        for i in range(2, len(sys.argv)):
            if sys.argv[i].split('=')[0] == 'number_of_folds':
                number_of_folds = int(sys.argv[i].split('=')[1])
            if sys.argv[i].split('=')[0] == 'categorical_column_handling_method':
                categorical_column_handling_method = str(sys.argv[i].split('=')[1])

        lgbm.classify_and_report(crime_df, number_of_folds, categorical_column_handling_method)

    elif mode == conf.VISUALIZE_MODE:
        crime_df = utils.read_dataset(preprocessed_crime_dataset_file_path)
        while True:
            txt = input("Please choose the type of visualizer (pie, line) or q to quit: ")
            if txt not in constants.VISUALIZATION_INPUTS[0]:
                print('Invalid Visualizer!')
                continue
            if txt == 'pie':
                while True:
                    column_name = input("Please choose the column name: ")
                    if column_name not in constants.VISUALIZATION_INPUTS[1]:
                        print('Invalid Column!')
                        continue
                    visualizer.construct_pie_chart(crime_df, column_name)
                    break
            if txt == 'line':
                while True:
                    column_name = input("Please choose the column name: ")
                    if column_name not in constants.VISUALIZATION_INPUTS[1]:
                        print('Invalid Column!')
                        continue
                    title = input("Please choose the title name: ")
                    x_label = input("Please choose the x label name: ")
                    y_label = input("Please choose the y label name: ")
                    visualizer.construct_line_chart(crime_df, column_name, x_label, y_label, title)
                    break
            if txt == 'q':
                break
