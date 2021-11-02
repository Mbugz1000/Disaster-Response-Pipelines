import sys
import pandas as pd
from sqlalchemy import create_engine

home = '../../'
data_dir = home + '/data'
raw_dir = data_dir + '/raw'
processed_dir = data_dir + '/processed'
modelling_dir = data_dir + '/for_modelling'


def load_data(messages_filepath, categories_filepath):
    """
    This method loads and merges the messages and categories dataframes.

    :param messages_filepath:
    :param categories_filepath:
    :return: Merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on=['id'], how='inner')

    return df


def clean_data(df):
    """
    In this method, I clean the data by taking the followins steps:
        1. Split categories into separate category columns
        2. Convert category values to just numbers 0 or 1
        3. Replace categories column in df with new category columns.
        4. Remove duplicates

    :param df: Merged Categories and Messages DataFrame
    :return: Cleaned Dataframe
    """

    # Split categories into separate category columns
    categories = df.categories.str.split(';', expand=True)

    row = categories.iloc[0]
    category_colnames = list(row.str.extract('([\w\W]+)-[\d]$', expand=False).values)
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Replace categories column in df with new category columns.
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)

    # Dropping Duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    This method saves a dataframe to an SqlLite DB
    :param df:
    :param database_filename:
    :return: None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('CategorisedMessages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()