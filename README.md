# Crime Data Analysis
This Data Mining project aims to analyze crime datasets and  classify type of crime that is likely to occur given 
circumstances. This project is carried out as term project for CEng514 / Data Mining graduate course in 2020 Spring
semester at Computer Engineering department of Middle East Technical University. More details on project results
and discussion can be seen in resources/Ceng514_DataMining_ProjectFinalReport.pdf

## Dataset
Mainly, [Los Angeles Crime dataset](https://www.kaggle.com/cityofLA/crime-in-los-angeles) is used to make 
classification. In addition to this, we utilized 
[Los Angeles Census dataset](https://www.kaggle.com/cityofLA/los-angeles-census-data) to see whether our model 
would improve. More details on dataset statistics and discussion on features can be seen in project final report.
Additionally, to correlate these datasets, conversion from latitude longitude information
in crime dataset to zip code information in census dataset is conducted. Zip code to
geolocation map for Los Angeles is taken from [simplemaps](https://simplemaps.com/data/us-zips).


## Usage
Pyhton 3.9 and Conda environment with dependencies as given in requirements.txt is used. Program
expects each dataset and zip code to geolocation map in data/raw folder. You can download csv files from the given
links in Dataset section and place them. 

To run the main.py, you need to specify mode as first argument. For each mode, there are additional optional 
parameters which all have default values, but you can change them according to your needs. For more information 
on optional parameters, see main.py file. Following modes can be used:

- Run preprocessing steps on crime dataset and save resulting csv

`python3 main.py preprocess_crime <additional optional parameters>`

- Run preprocessing steps on census dataset and save resulting csv

`python3 main.py preprocess_census <additional optional parameters>`

- Correlate census and crime datasets using zip codes and add features
from census dataset into crime dataset, then save resulting crime dataset

`python3 main.py correlate <additional optional parameters>`

`python3 main.py merge_datasets <additional optional parameters>`

- Use Categorical Naive Bayes to classify crime codes

`python3 main.py classify_cnb <additional optional parameters>`

- Use Light Gradient Boosting Machine to classify crime codes

`python3 main.py classify_lgbm <additional optional parameters>`

- Use K-Nearest Neighbor to classify crime codes

`python3 main.py classify_knn <additional optional parameters>`

- Visualize datasets

`python3 main.py visualize <additional optional parameters>`

## License
[MIT](https://choosealicense.com/licenses/mit/)
