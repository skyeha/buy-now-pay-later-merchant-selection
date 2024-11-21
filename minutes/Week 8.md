# Buy Now, Pay Later - Industry Group 9
10/09/2024

# Group Members:
* Alistair Wern Hao Cheah
* Do Nhat Anh Ha (Skye)
* Shiping Song
* Sitao Qin (Alex)

# Agenda:
* **Using datetime as a preditor for predicting fraud probability for both mechants and consumer**: Deciding on splitting datetime into day (e.g Monday - Saturday) and month. They will all be encoded using a type of encoding that can preserve their ordinal structure. We will also create features that determine if that day is a weekend or weekday and if it's a holiday or not.
* **Fraud probability for transactions**: Discuss about cases when 1 consumer, has a fraud probability, have multiple purchases in the same day. We agreed that if that is the case then those transactions will have the same fraud probability as we assumed that if a reasonal consumer will scam on multiple purchases in the same day instead of scamming only one.
* **Feature engineering**: Discussed about potential features that can be used. For instance, determine how much purchases a customer made 1 week before they make their new transaction. Another one the proportion of the money spend on shopping in the median/mean income of the LGA region that the consmer is living in. If it surpasses a certain threshold, it's likely a scam.

# Task Allocation:
## Individual
* Skye:
    * Finalise everyone's work and make adjustment where needed.
    * Outlier removal
* Alex: 
    * Create additional feature to train for consumer's fraud probability predicting model

## Group work
* Shiping and Alistair: 
    * Model on predicting fraud probability for merchants.

**Note**: use graphs/drawings/diagrams whenever possible so that they can be used in the future for the presentation. This will help the audience understand our approach as well as something we can refer back when lost. 



