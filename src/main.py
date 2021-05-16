import crime_preprocessor as ud
import census_preprocessor as cp
import dataset_correlation as dc
import crime_classification_lgbm as cc
import crime_classification_categorical_nb as cnb
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    crime_dataset_file_name = 'Crime_Data_2010_2017.csv'
    preprocessed_crime_dataset_file_name = 'crime_data_2010_2017_preprocessed.csv'
    preprocessed_crime_dataset_file_name_ohe = 'crime_data_2010_2017_preprocessed_new.csv'
    correlated_crime_dataset_file_name = 'crime_data_2010_2017_correlated.csv'

    census_dataset_file_name = '2010-census-populations-by-zip-code.csv'
    preprocessed_census_dataset_file_name = '2010-census-populations-by-zip-code-preprocessed.csv'

    # ud.preprocess_and_save(crime_dataset_file_name, preprocessed_crime_dataset_file_name)
    # cp.preprocess_and_save(census_dataset_file_name, preprocessed_census_dataset_file_name)

    df = ud.read_dataset(preprocessed_crime_dataset_file_name)
    # census_df = ud.read_dataset(preprocessed_census_dataset_file_name)

    # dc.correlate_and_save(df, census_df, correlated_crime_dataset_file_name)

    ####################################################################################################################
    # for the solution with one hot encoding, please read the dataset named preprocessed_crime_dataset_file_name_ohe
    # column_names = ['TimeOccurred', 'VictimAgeStage', 'VictimSex', 'VictimDescent', 'SeasonOccurred', 'DayOfWeek']
    # df = cc.obtain_ohe_df(df, column_names)

    # for the solution with label encoder, please read the dataset named preprocessed_crime_dataset_file_name
    # df[['VictimSex', 'VictimDescent']] = df[['VictimSex', 'VictimDescent']].apply(LabelEncoder().fit_transform)

    # common part for both label encoding and one hot encoding
    # experiments = cc.get_experiments(df)

    # cc.print_error(cc.calculate_error(cc.predict_value(cc.construct_model_with_decision_tree(experiments[0][0]), experiments[0][1])),
    #                cc.calculate_error(cc.predict_value(cc.construct_model_with_decision_tree(experiments[1][0]), experiments[1][1])),
    #                cc.calculate_error(cc.predict_value(cc.construct_model_with_decision_tree(experiments[2][0]), experiments[2][1])))
    ####################################################################################################################

    # cnb.classify_and_report(df, 3)
