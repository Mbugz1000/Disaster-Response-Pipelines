# Disaster Response Pipelines
## Installation
Run the following command to set up the environment for this project. 

```
conda create --name project_2_udacity python==3.7.3 -y && conda activate project_2_udacity && conda install pandas jupyterlab altair "pandas-profiling>2.0" scikit-learn flask nltk sqlalchemy joblib -y
```

## Project Motivation
This is the second of several Udacity projects in the Data Science Nano-degree program. The primary aim of this project 
is therefore to complete the course-work of the mentioned program. 


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
   - To run ETL pipeline that cleans data and stores in database  
     `python src/deployment/process_data.py data/raw/messages.csv data/raw/categories.csv data/processed/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves (NB: Training takes +1hr)  
     `python src/deployment/train_classifier.py data/processed/DisasterResponse.db data/for_modelling/classifier.pkl`
2. Fetch the model classifier from [here](https://drive.google.com/file/d/16B1C-Uso2A0eopSyLK3YJOLT2t3NDLhs/view?usp=sharing). (NB: The classifier is about 1.5 GB, so it could not be included in the repository)
2. Run the following command in the app's directory to run your web app.  
   `python src/deployment/app/run.py`

3. Go to http://127.0.0.0:3001/

## Results Summary
- Best Performance that was achieved using Grid Search:  
    ```
    {'accuracy': 0.9650076277650648, 'precision': 0.75, 'f1score': 0.22821576763485477}
  ```


## Licensing, Authors & Acknowledgements
- [Udacity DataScience Nano Degree](https://www.udacity.com/course/data-scientist-nanodegree--nd025?gclid=Cj0KCQjw5oiMBhDtARIsAJi0qk2AQs-eBmFS3MzXbZwJRVcgx36bu9tZls_UXsTxki-oYNOcuYvkqfsaAv-dEALw_wcB&utm_campaign=12908932988_c&utm_keyword=%2Budacity%20%2Bdata%20%2Bscience_b&utm_medium=ads_r&utm_source=gsem_brand&utm_term=124509203711)

