# Disaster Response Pipelines
## Installations
Run the following command to set up the environment for this project. 

```
conda create --name project_2_udacity python==3.7.3 -y && conda activate project_2_udacity && conda install pandas jupyterlab altair "pandas-profiling>2.0" scikit-learn flask -y
```




## Project Motivation
This is the second of several Udacity projects in the Data Science Nano-degree program. The primary aim of this project 
is therefore to complete the course-work of the mentioned program. 

1. 

## File Descriptions
This project is has 2 primary folders: 
- **data** : Data Folder. It contains 3 folders
    - **raw**: 
    - **processed**: 
    - **for_modelling**: 
- **src**: Source code folder. 
    - 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
   `python run.py`

3. Go to http://0.0.0.0:3001/

## Results Summary
- e

## Licensing, Authors & Acknowledgements
- w

