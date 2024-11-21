# Buy Now, Pay Later - Industry Group 9
05:09 PM - 03/09/2024

# Group Members:
* Alistair Wern Hao Cheah
* Do Nhat Anh Ha (Skye)
* Shiping Song
* Sitao Qin (Alex)

# Agenda:
* **Null values for consumer/merchant fraud probability** in certain transactions. We decide that we will create 2 prediction models that will predict the fraud probability for consumer/merchant for transaction that does not have these values previously.
* **Visualisation for preliminary analysis**: Pie chart or any kind of graphs that are suitable. For instance, for `revenue_level = "a"`,  show a pie chart of merchants that make up that revenue (potential show top 5/10/15/20)
* **Outlier analysis**: use statistical method to restrict extreme value, leaning towards a modified version of Winsorizing.
* **External dataset**: dataset on demographics and socio-economy of each state were found and will be use as part of the modeling

# Task Allocation:
## Individual
* Skye:
    * ETL Pipeline for external datasets and outlier analysis. 
    * Write report about missing/ null values after collecting them from Shiping and Alistair
    * Finalise the general ETL. Work on exporting the cleaned data to the curated folder.
* Alex: 
    * Finalise any geospatial analysis

## Group work
* Shiping and Alistair: 
    * Summaries on null values in `consumer_fraud` and `merchant_fraud` when merging the original datasets.
    * Create charts (as outlined above) for future uses in presentation.
* Alistair and Alex:
    * Plan and document on how to generate simple model for predicting `customer_fraud`. 
* Shiping and Skye:
    * Plan and document on how developing a simple model for predicting `merchant_fraud`.

**Note**: use graphs/drawings/diagrams whenever possible so that they can be used in the future for the presentation. This will help the audience understand our approach as well as something we can refer back when lost. 



