# Buy Now Pay Later: Merchant Selection
## Project Overview
This project is the second assignment (group) of the subject Applied Data Science (MAST30034) at The University of Melbourne. The goal of this project is to create a ranking system that helps a Buy Now Pay Later firm (i.e. AfterPay) select top 100 merchants to partner up. 

Data regarding customer information, merchant information, customer's default rate, merchant's default rate, and transactions were provided by the firm. External datasets were employed to aid our analysis as well as ML models. Majority of external datasets were found mostly from Australia Beaurau of Statistic's website.

## Our Approach: A Summary
Having multiple datasets encourage us to employed an ETL pipeline that help us automate and transform the data into desirable format that can be used. The pipeline also include basic data cleaning (i.e. null entries removal), impute missing value using unsupervised learning whenever possible, duplicate removals, and datasets merging.

Preliminary and geospatial analysis were conducted for better understanding of datasets, especially data regarding default rate of both customer and merchants. This was crucial in determining our approach on how we generate our model to predict these default rate since only some customer and merchant have default rate for model training. We employed 2 models for both category, namely Linear Regression (baseline model) and Random Forrest Regression.

We decided to rank the merchant by treating each merchant as an investment that generates profit the BNPL firm, which ever merchants have higher profitability are more favorable. We employed **Discounted Cash Flow** model, a concept in finance that evaluates value of an investment project by perform certain calculation on their forecasted revenue. We adopted 2 methods to predict the revenue: 

1. Compute average monthly growth rate and assumed revenue grow by the same rate for each month
2. Using a Long Short Term Neural Network

The further we forecast into the future, the less accurate. We decided to only forecast 3 months ahead. Once we got the base value of each merchant, adjustment and penalty terms are applied. The adjustments and penalty terms are based on the characteristics of merchants such as: growth stability, order volume stability, associated default risk, etc. (further detail contained in notebook). We further splitted the merchants into 5 different segments using simple NLP, aiming to reduce BNPL's firm chance on relying on only one segments.

Overall, we created a ranking system that was insightful and interpretable for non-technical stakeholders.

## Timeline
The project tooks 6 week to complete. The timeline is as follow:
1. **Week 1**: data collection, ETL pipeline constrution, preliminary anaysis ~ 15 hours
2. **Week 2**: outlier analysis, geospatial analysis, visualisation ~ 17 hours
3. **Week 3**: Modelling Default Rate ~ 14 hours
4. **Week 4**: Features and Heursitics Engineer for Ranking System ~ 14 hours
5. **Week 5**: Segmenting merchants ~ 3 hours
6. **Week 6**: Finalise ranking system and identify top 100 merchants ~ 12 hours

## Presentation
Click [here](https://www.canva.com/design/DAGSUw-gtaE/JNJhYQR_RFOx6pm2KY7fGQ/edit?utm_content=DAGSUw-gtaE&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) to access presentation slides for this project that we present to stakeholders (i.e subject coordinator, tutor, and peers)

## Recreating The Project Instruction
Inital setup:
1. Clone the repo
2. Download the BNPL dataset from [here](https://drive.google.com/drive/folders/1-7271NwUF4oLpy6wVx4D9O8mU0E-Ue4k?usp=sharing)

After setting up the repo, run each notebook in the `notebook` directory in order.
