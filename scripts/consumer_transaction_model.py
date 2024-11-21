import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

from pyspark.sql import functions as F, SparkSession
from pyspark.sql.window import Window
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array

def feature_visualisation(df_pandas, plots):
    """ 
    Create a list of plots for feature visualisation, including bar charts, scatter plot and histogram.
    """
    # Set up a big plot grid
    fig, axes = plt.subplots(6, 3, figsize=(20, 20))  # 4x3 grid of subplots
    fig.tight_layout(pad=5.0)  # Adjust spacing between plots

    for i, (plot_title, (feature, plot_type)) in enumerate(plots.items()):
        ax = axes[i // 3, i % 3]  # Select subplot position

        if plot_type == "hist":
            sns.histplot(df_pandas[feature], bins=30, kde=True, ax=ax)
        elif plot_type == "count":
            sns.countplot(x=feature, data=df_pandas, ax=ax)
        elif plot_type.startswith("scatter"):
            if plot_type == "scatter1":
                sns.scatterplot(x="Proportion_between_max_order_value_mean_income", y="fraud_probability", data=df_pandas, ax=ax)
            elif plot_type == "scatter2":
                sns.scatterplot(x="Proportion_between_max_order_value_median_income", y="fraud_probability", data=df_pandas, ax=ax)
            elif plot_type == "scatter3":
                sns.scatterplot(x="Proportion_between_total_order_value_mean_income", y="fraud_probability", data=df_pandas, ax=ax)
            elif plot_type == "scatter4":
                sns.scatterplot(x="Proportion_between_total_order_value_median_income", y="fraud_probability", data=df_pandas, ax=ax)

        ax.set_title(plot_title)

    # Show the entire plot
    plt.show()
    
def assemble_data(data, predictors):
    # Apply log transformation to dollar value related columns
    cols_to_log = ['dollar_value', 'average_dollar_value', 'min_dollar_value',
                'max_dollar_value', 'stddev_dollar_value',
                "Proportion_between_max_order_value_mean_income",
                "Proportion_between_max_order_value_median_income",
                "Proportion_between_total_order_value_mean_income",
                "Proportion_between_total_order_value_median_income"
                ] 
    
    for col in cols_to_log:
        data = data.withColumn(col, F.when(F.col(col) > 0, F.log(F.col(col))).otherwise(None))

    
    # Apply normalisation to columns of choice
    cols_to_norm = ["dollar_value", "min_dollar_value", "max_dollar_value", "stddev_dollar_value","average_dollar_value"]
    
    for col in cols_to_norm:
        norm_assembler = VectorAssembler(inputCols=[col], outputCol= f'{col}_vec')
        data = norm_assembler.transform(data)
        scaler = StandardScaler(inputCol=f"{col}_vec", outputCol=f"norm_{col}")
        data = scaler.fit(data.select(f"{col}_vec")).transform(data)
        
    
    # Apply StringIndexing to month to keep ordinal structure
    month_indexer = StringIndexer(inputCol='month', outputCol='month_index')
    weekday_indexer = StringIndexer(inputCol='day_of_week', outputCol='weekday_index')
    
    is_weekend_vector = OneHotEncoder(inputCol='is_weekend', outputCol='is_weekend_vector')
    
    assembler = VectorAssembler(inputCols=predictors, outputCol='features')
    pipeline = Pipeline(stages=[month_indexer, weekday_indexer, is_weekend_vector, assembler])
    
    assembled_data = pipeline.fit(data).transform(data)
    
    return assembled_data, assembler
    
