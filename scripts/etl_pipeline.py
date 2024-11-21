from pyspark.sql import functions as F, SparkSession, DataFrame
from pyspark.sql.types import IntegerType, LongType, DoubleType, StringType, DoubleType
from functools import reduce
from pyspark.sql.functions import col, sum
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from urllib.request import urlretrieve
import zipfile
import os

pd.options.mode.chained_assignment = None 

def replace_id(map_df, target_df):
    """
        Replace all user_id by consumer_id
    """
    mapped_df = target_df.join(map_df, on="user_id", how="inner")
    mapped_df = mapped_df.drop('user_id')
    
    return mapped_df

def clean_merchant_details(merchant_df):
    
    """
        This function takes in a merchants dataset and transforms/cleans it into a suitable format.
        Also indicates dataset size before and after. Returns cleaned dataset.
    """

    # Get dataset size before cleaning
    print("Before: ")
    get_dataset_count(merchant_df)

    merchant_df = merchant_df.withColumn("tags", F.regexp_replace("tags", r"^[\(\[]|[\)\]]$", "")) # Remove the outermost bracket
    merchant_df = merchant_df.withColumn("tags", F.regexp_replace("tags", r"[\)\]],\s*[\(\[]", r")\|(")) # Replacing the comma that separate each tuple/list into "|"

    # Split accordingly 
    merchant_df = merchant_df.withColumn("tags", F.split("tags", "\|")) 
    merchant_df = merchant_df.withColumns({"category": F.regexp_replace(F.col("tags").getItem(0), r"^[\(\[]|[\)\]]$", ""),
                         "revenue_level": F.regexp_replace(F.col("tags").getItem(1), r"^[\(\[]|[\)\]]$", ""),
                         "take_rate": F.regexp_extract(F.col("tags").getItem(2), r"take rate: (\d+\.\d+)",1).cast(DoubleType())
                        })
    
    # Make it consistently lower case
    merchant_df = merchant_df.withColumn("category", F.lower(F.col("category")))

    # Drop original feature column (not needed anymore)
    merchant_df = merchant_df.drop("tags")

    # Ensure revenue level is within a defined range of (a-e)
    merchant_df = merchant_df.filter((F.col("revenue_level") == "a") | (F.col("revenue_level") == "b") | (F.col("revenue_level") == "c") |
                   (F.col("revenue_level") == "d") | (F.col("revenue_level") == "e"))
    

    # Ensure take_rate is within a defined range (0.0 to 100.0)
    merchant_df = merchant_df.filter((F.col("take_rate") >= 0.0) & (F.col("take_rate") <= 100.0))

    # Get dataset size after cleaning
    print("After: ")
    get_dataset_count(merchant_df)

    return merchant_df


def clean_consumer_details(consumer_df):
    """
        This function takes in the consumer dataset, and restructures the data into a suitable format.
        Also indicates dataset size before and after. Returns cleaned dataset.
    """

    # Get dataset size before cleaning
    print("Before: ")
    get_dataset_count(consumer_df)


    column_name = str(consumer_df.columns[0])
    consumer_df = consumer_df.withColumn("name", F.split(consumer_df[column_name], r'\|').getItem(0)) \
                .withColumn("consumer_id", F.split(consumer_df[column_name], r'\|').getItem(5))\
                .withColumn("gender", F.split(consumer_df[column_name], r'\|').getItem(4))\
                .withColumn("state", F.split(consumer_df[column_name], r'\|').getItem(2)) \
                .withColumn("postcode", F.split(consumer_df[column_name], r'\|').getItem(3)) \
            

    consumer_df = consumer_df.withColumn("postcode", F.col("postcode").cast(IntegerType())) \
                .withColumn("consumer_id", F.col("consumer_id").cast(LongType()))
    consumer_df = consumer_df.drop(column_name)

    # Get dataset size after cleaning
    print("After: ")
    get_dataset_count(consumer_df)

    return consumer_df

def get_dataset_count(df):

    """
        This function takes in a dataset and prints its count. (size) 
    """

    count = df.count()
    print("The dataset count is ", count )

    return

def ensure_datetime_range(df, start, end):
    """
        This function ensures that a dataframe with a column that specifies datetime is within the desire datetime range
    """
    inital_entries = df.count()
    df = df.filter((start <= F.to_date(F.col("order_datetime"))) &
                           (F.to_date(F.col("order_datetime")) <= end))
    
    final_entries = df.count()
    print(f"Starting entries: {inital_entries} \nFinal entries: {final_entries}")
    print(f"Net change (%): {round((inital_entries - final_entries)/inital_entries * 100, 2)} ")
    return df

def calculate_missing_values(df):
    """
    Takes in a DataFrame, calculates the count of missing (NULL) values 
    for each column, and displays it as a table.
    """

    # Initialise an empty list to hold the expressions for summing NULL counts
    null_count_columns = []

    # Iterate over each column, summing their counts of NULL values
    for column in df.columns:
        null_count_expression = F.sum(F.col(column).isNull().cast("int")).alias(column + '_missing_count')
        null_count_columns.append(null_count_expression)

    # Select the summed NULL counts and display them
    missing_value_counts = df.select(null_count_columns)
    missing_value_counts.show()

    return

def clean_postcode_lga_mapping(df):
    """
        This funcion clean the data on mapping postcode to LGA as well as impute missing LGA code using KNN
    """

    cols = ["postcode", "state", "long","lat", "lgacode"]
    df = df[cols]
    df.drop_duplicates(subset=["postcode"], inplace=True) # dropping any duplicates 


    # Separate entries with LGA code and entries without LGA code
    df_missing = df[df['lgacode'].isnull()]
    df_not_missing = df.dropna(subset=['lgacode'])

    # Split predictor and lable for classification
    X_train = df_not_missing[['long', 'lat']]
    y_train = df_not_missing['lgacode']

    # Intialise the model
    knn = KNeighborsRegressor(n_neighbors=1)
    knn.fit(X_train, y_train)

    # Predict missing lgacode for entries with missing lgacode
    X_missing = df_missing[['long', 'lat']]
    df.loc[df['lgacode'].isnull(), 'lgacode'] = knn.predict(X_missing)

    df['lgacode'] = df['lgacode'].astype(int)

    return df

def preprocess_income_df(df):
    renamed_col = {'Unnamed: 0': 'lga', 'Unnamed: 1': 'lga_name', 'Earners': 'num_earners', 'Median age of earners': 'median_age',
               'Sum': 'total_income', 'Median': 'median_income', 'Mean': 'mean_income', 'Gini coefficient': 'gini_coef'}

    df = df.rename(columns=renamed_col)

    remove_rows = ['LGA', 'Australia ', 'New South Wales', 'Victoria', 'Queensland', 'South Australia', 'Western Australia', 
                   'Tasmania', 'Northern Territory', 'Australian Capital Territory']

    for row in remove_rows:
        df = df.drop(df[df['lga'] == row].index)
    
    df = df.reset_index()
    df = df.drop(columns = 'index')

    selected_cols = ['lga', 'median_age', 'median_income',
                     'mean_income']
    
    df = df[selected_cols]

    for index, row in df[df['median_age'] == 'np'].iterrows():
        df.loc[index, 'median_age'] = 42
        df.loc[index, 'median_income'] = "58,591"
        df.loc[index, 'mean_income'] = "75,878"

    for col in selected_cols[2:]:   
        df[col] = df[col].str.replace(",", "").astype(int)

    return df

def impute_income_metrics(df):
    missing = df[df.lga.isnull()]
    not_missing = df.dropna(subset=['lga'])

    X_train = not_missing[['long', 'lat']]
    y_train = not_missing[['median_age', 'median_income', 'mean_income']]

    knn = KNeighborsRegressor(n_neighbors=1)
    knn.fit(X_train, y_train)

    X_missing = missing[['long', 'lat']]
    df.loc[df.lga.isnull(),['median_age', 'median_income', 'mean_income']] = knn.predict(X_missing)

    return df

def process_fp_data(path):
    columns_name = {
        "Table 4a" : {'Unnamed: 0': 'state', 'Unnamed: 7': "victimisation_rate"},
        "Table 4b" : {'Unnamed: 0': 'state', 'Unnamed: 7': "rse_percent"}
    }

    state_mapping = {
        "New South Wales" : "NSW", "Victoria": "VIC", "Queensland": "QLD",
        "South Australia": "SA", "Western Australia": "WA", "Tasmania": "TAS",
        "Northern Territory": "NT", "Australian Capital Territory": "ACT"
    }

    # Loading in the table 
    df_1 = pd.read_excel(io = path, sheet_name="Table 4a", skiprows = 8, skipfooter= 70, usecols = "A,H")
    df_2 = pd.read_excel(io = path, sheet_name="Table 4b", skiprows = 8, skipfooter= 67, usecols = "A,H")

    # Renaming the columns
    df_1 = df_1.rename(columns = columns_name["Table 4a"])
    df_2 = df_2.rename(columns = columns_name["Table 4b"])
    
    # Reduce that state name into code
    for index , row in df_1.iterrows():
        df_1.loc[index, "state"] = state_mapping[row['state']]
        df_2.loc[index, "state"] = state_mapping[row['state']]

    df = df_1.merge(df_2, on="state", how = "inner")

    return df