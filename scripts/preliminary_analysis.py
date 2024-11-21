from pyspark.sql import functions as F, SparkSession
from pyspark.sql.types import IntegerType, LongType, DoubleType, StringType, DoubleType
from pyspark.sql.types import DoubleType, FloatType, DateType, StringType
from pyspark.sql import functions as F


def get_dataset_count(df):

    """
        This function takes in a dataset and prints its count. (size) 
    """

    count = df.count()
    print("The dataset count is ", count )

    return


def calculate_missing_values(df):
    """
    Takes in a DataFrame, calculates the count of missing (NULL or NaN) values 
    for each column, and displays it as a table.
    """

    # Initialise an empty list to hold the expressions for summing missing value counts
    missing_count_columns = []

    # Iterate over each column, summing their counts of NULL or NaN values
    for column in df.columns:

        column_type = df.schema[column].dataType
        
        # For numeric columns, check both NULL and NaN values
        if isinstance(column_type, (DoubleType, FloatType)):
            missing_count_expression = F.sum(
                (F.col(column).isNull() | F.isnan(F.col(column))).cast("int")
            ).alias(f"{column}_missing_count")
        
        # For non-numeric columns, only check for NULL values
        else:
            missing_count_expression = F.sum(
                F.col(column).isNull().cast("int")
            ).alias(f"{column}_missing_count")
        
        missing_count_columns.append(missing_count_expression)


    # Select the summed NULL and NaN counts and display them
    missing_value_counts = df.select(missing_count_columns)
    missing_value_counts.show()

    return