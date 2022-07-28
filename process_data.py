# Import libraries and packages
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import warnings
warnings.filterwarnings('ignore')

def load_data(messages_filepath, categories_filepath):
    '''loads the message and categories dataframes
    Returns:
    dataframe (df) containing messages_filepath and categories_filepath
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge datasets
    df = pd.merge(messages, categories, on="id")
    df.head()
    return df


def clean_data(df):
    """
    This function cleans the df.
    """
    # Split `categories` into separate category columns.
    categories = df.categories.str.split(";", expand=True)
    categories.head()
    # select the first row of the categories dataframe
    # select the first row of the categories dataframe
    row = categories.head(1)
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :]
    print(category_colnames)
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()
    # iterate through the category columns in df to keep only the
    # last character of the string
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-").str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    categories.head()
    # replace 2s with 1s in related column
    categories['related'] = categories['related'].replace(to_replace=2, value=1)
    # drop the original categories column from `df`
    df = df.drop(["categories"], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    df.head()
    # check number of duplicates
    df.duplicated().sum()
    # drop duplicates
    df = df.drop_duplicates()
    # check number of duplicates
    df.duplicated().sum()
    return df


def save_data(df, database_filename):
    """
    This function stores the df in a SQL database
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('pipelines', engine, index=False)


def main():
    """
        Data processing functions:
           1) Load Messages Data with Categories
           2) Clean Categories Data
           3) Save Data to SQLite Database
       """
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
