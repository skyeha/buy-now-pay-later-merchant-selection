# Buy Now, Pay Later - Industry Group 9
05:09 PM - 29/8/2024

# Group Members:
* Alistair Wern Hao Cheah
* Do Nhat Anh Ha
* Shiping Song
* Sitao Qin

# Agenda:
* **Consumer/merchant fraud probability**: found out that fraud probability for merchant and consumer are for certain days only, not the fraud probability of a customer or a merchant. Decided to leave the fraud probability tables alone for now and yet to merge it with other table
* **External dataset**: discussion on prospective external dataset, *Demographics* and *Shapefile* of each postcode might be beneficial for analysis. 
* **`user_id`, `consumer_id`**: all members agreed upon using `consumer_id` as the sole key
* **Preliminary analysis**: decided to do analysis on 3 areas:
    1. Using data on transactions records, group rows by ABN and find total revenue and average dollar per order of that merchant. Finding which merchant has the highest sales
    2. Add an additional column to the transactions records, indicating which revenue level the merchant belongs to. Thus, find the total revenue of each revenue class. Additional analysis may include which merchant(s) make up most of the revenue for their respective level.
    3. Find the average `take_rate` for each revenue level
    4. Using data on consumer, count the number of consumer for each postcode. Using external data on the shapefile of each postcode, map the distribution of consumer across postcode.

# Task Allocation:
* Alistair and Shiping: Work together on carrying out the preliminary analysis, bullet points **1-3**
* Sitao: carrying out the task detailed in bullet point **4** of the preliminary analysis
* Do Nhat Anh Ha: exploration of external dataset
* Sitao & Do Nhat Anh Ha: Work together on the ELT Pipeline

# Signed: 
* Alistair Wern Hao Cheah
* Do Nhat Anh Ha
* Shiping Song
* Sitao Qin