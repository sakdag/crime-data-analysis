# Original crime dataset column names
LOCATION = 'Location'
DATE_OCCURRED = 'DateOccurred'
CRIME_CODE = 'CrimeCode'
VICTIM_SEX = 'VictimSex'
VICTIM_DESCENT = 'VictimDescent'
PREMISE_CODE = 'PremiseCode'
VICTIM_AGE = 'VictimAge'
CRIME_CODE_DESCRIPTION = 'CrimeCodeDescription'
DR_NUMBER = 'DRNumber'
PREMISE_DESCRIPTION = 'PremiseDescription'

# Created crime dataset column names
LATITUDE = 'Latitude'
LONGITUDE = 'Longitude'
TIME_OCCURRED = 'TimeOccurred'
MONTH_OCCURRED = 'MonthOccurred'
DAY_OF_WEEK = 'DayOfWeek'
VICTIM_AGE_STAGE = 'VictimAgeStage'
SEASON_OCCURRED = 'SeasonOccurred'

# List of unused column names in crime dataset
UNUSED_COLUMN_LIST = {
    'DateReported', 'AreaID', 'AreaName',
    'ReportingDistrict', 'MOCodes', 'WeaponUsedCode',
    'WeaponDescription', 'StatusCode', 'StatusDescription',
    'CrimeCode1', 'CrimeCode2', 'CrimeCode3', 'CrimeCode4',
    'Address', 'CrossStreet'
}

# Census dataset column names
ZIP_CODE = 'ZipCode'
TOTAL_POPULATION = 'TotalPopulation'
MEDIAN_AGE = 'MedianAge'
TOTAL_MALES = 'TotalMales'
TOTAL_FEMALES = 'TotalFemales'
TOTAL_HOUSEHOLDS = 'TotalHouseholds'
AVERAGE_HOUSEHOLD_SIZE = 'AverageHouseholdSize'
