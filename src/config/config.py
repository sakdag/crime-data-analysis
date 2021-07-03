# Application modes:
PREPROCESS_CRIME_DATASET_MODE = 'preprocess_crime'
PREPROCESS_CENSUS_DATASET_MODE = 'preprocess_census'
CORRELATE_CRIME_DATASET_MODE = 'correlate'
CLASSIFY_WITH_CATEGORICAL_NB_MODE = 'classify_cnb'
CLASSIFY_WITH_LGBM_MODE = 'classify_lgbm'
VISUALIZE_MODE = 'visualize'

# Original file paths
CRIME_DATASET_FILE_PATH = 'data/Crime_Data_2010_2017.csv'
CENSUS_DATASET_FILE_PATH = 'data/2010-census-populations-by-zip-code.csv'

# Preprocessed crime dataset file paths
PREPROCESSED_CRIME_DATASET_FILE_PATH = 'data/crime_data_2010_2017_preprocessed.csv'
PREPROCESSED_CRIME_DATASET_FILE_PATH_WO_NAN = 'data/crime_data_2010_2017_preprocessed_drop_nan.csv'
PREPROCESSED_CRIME_DATASET_FILE_PATH_OHE = '../crime_data_2010_2017_preprocessed_ole.csv'
CORRELATED_CRIME_DATASET_FILE_PATH = 'data/crime_data_2010_2017_correlated.csv'

# Preprocessed census dataset file paths
PREPROCESSED_CENSUS_DATASET_FILE_PATH = 'data/2010-census-populations-by-zip-code-preprocessed.csv'

# Additional dataset file paths
ZIP_CODES_DATASET_FILE_PATH = 'data/uszips.csv'
