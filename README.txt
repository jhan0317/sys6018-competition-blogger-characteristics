
# MEMBERS
Beni Shpringer
Jiangxue Han
Nicola Bruno

# GOALS
Predict the age of blog's authors

Have at MINIMUM one parametric model:
1. Ridge regression
2. Lasso regression

Implement a cross-validation approach:
1. k-fold cross validation

# Bonus
Get a good score on the parametric linear model modeling for bonus credit

# RESULTS
Ridge regression: 3.87204

# FILES
Code: All of the code.
Data: All of the cleaned data.
Final_submission: All of the final submissions for Kaggle
Large-size datasets: Some datasets are too large to be uploaded to github and are stored in Google Drive.
                     https://drive.google.com/open?id=1M1bk-_DDEJxfOb7HrCpxgM712CoAm9zU

# Code
# initial_data_analysis.py: Reads in the raw dataset and finds trends in the text data, specifically in relation to the 'age' variable
# data_cleaning.py: Reads in the raw dataset and conducts data prepocessing
# text_mining.py: Reads in the raw dataset, aggregates the dataset and conducts basic text cleaning
# tfidf.R: Reads in the cleaned text and calculates the term frequency-inverse document frequency
# linear_model.py: Reads in the cleaned data and tfidf data. Builds the final model based off of analysis conducted in other .py files


# Final_submission
# blog_result_1009_V4.csv: the result with best score
