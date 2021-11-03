# Disaster Response Pipelines
## Installations
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
   - To run ML pipeline that trains classifier and saves
     `python src/deployment/train_classifier.py data/processed/DisasterResponse.db data/for_modelling/model_output.pkl`
2. Run the following command in the app's directory to run your web app.
   `python src/deployment/app/run.py`

3. Go to http://127.0.0.0:3001/

## Results Summary
- e

## Licensing, Authors & Acknowledgements
- w

