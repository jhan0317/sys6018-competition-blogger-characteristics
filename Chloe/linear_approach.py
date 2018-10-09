# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV

# Read in raw datasets
raw_train = pd.read_csv('train.csv')
raw_test = pd.read_csv('test.csv')

# Get age of distinct user.ids
y_train = raw_train[['user.id','age']].drop_duplicates()
y_train = y_train.drop(['user.id'],axis=1)
y_train = y_train.reset_index(drop=True)

# Log transform the highly-skewed age
y_train = np.log(y_train)

x_train = raw_train.drop(['age'],axis=1)
x_test = raw_test

# Combine the training and testing set together
all_data = pd.concat([x_train,x_test],ignore_index=True)
all_data.columns
# Index(['user.id', 'gender', 'topic', 'sign', 'date', 'text'], dtype='object')

# Since post.id is useless in this case, we drop it
all_data = all_data.drop(['post.id'],axis=1)  

# Based on previous analysis, "year" is useful in prediction
all_data['date_split'] = all_data['date'].str.split(',')
all_data['year'] = [all_data['date_split'][x][2] for x in range(all_data.shape[0])]

# Store the information with distinct user.ids in another dataset
all_data_unique = all_data.iloc[:,0:4].drop_duplicates()
all_data_unique = all_data_unique.reset_index(drop=True)

# Create a variable to store the sentiment score for each user's blogs
all_data_unique['senti'] = 0   

# Create a variable to store the frequency of the year associated with blogs
all_data_unique['year_1999'] = 0
all_data_unique['year_2000'] = 0
all_data_unique['year_2001'] = 0
all_data_unique['year_2002'] = 0
all_data_unique['year_2003'] = 0
all_data_unique['year_2004'] = 0
all_data_unique['year_2005'] = 0
all_data_unique['year_2006'] = 0

i = 0
for id in all_data_unique['user.id']:
    # aggregate the texts of the same user.id
    texts = all_data[all_data['user.id'] == id][['text']] 
    values = texts.values.tolist()
    sen = ''.join(str(value) for value in values)
    # Calculates the sentiment score for each user
    blob = TextBlob(sen)
    all_data_unique.iloc[i,12] = blob.sentiment.polarity
    temp = all_data[all_data['user.id'] == id][['year']]
    for j in range(len(temp)):
        if (temp.iloc[j,] == "1999").bool() == True:
            all_data_unique.iloc[i,4] += 1
        elif (temp.iloc[j,] == "2000").bool() == True:
            all_data_unique.iloc[i,5] += 1
        elif (temp.iloc[j,] == "2001").bool() == True:
            all_data_unique.iloc[i,6] += 1
        elif (temp.iloc[j,] == "2002").bool() == True:
            all_data_unique.iloc[i,7] += 1
        elif (temp.iloc[j,] == "2003").bool() == True:
            all_data_unique.iloc[i,8] += 1
        elif (temp.iloc[j,] == "2004").bool() == True:
            all_data_unique.iloc[i,9] += 1
        elif (temp.iloc[j,] == "2005").bool() == True:
            all_data_unique.iloc[i,10] += 1
        elif (temp.iloc[j,] == "2006").bool() == True:
            all_data_unique.iloc[i,11] += 1
    i += 1

# Convert the categorical data into dummy variables
all_dummies = pd.get_dummies(all_data_unique)

all_dummies.shape
# (19320, 64)

# Read in the tfidf data that is pre-calculated in R
tfidf = pd.read_csv('Data/tfidf80.csv')
tfidf.rename(columns={'Unnamed: 0':'user.id'},inplace=True)

# Merge the tfidf data with cleaned dataset
complete_data = pd.merge(all_dummies,tfidf,on='user.id')
final_data = complete_data.drop(['user.id'],axis=1)

# Split the cleaned dataset to the training and test set
x_train = final_data.iloc[0:len(y_train),]
x_test = final_data.iloc[len(y_train):,]

# Validation
x_train_2, x_cv, y_train_2, y_cv = train_test_split(x_train, y_train,test_size=0.33, random_state=42)

# Implement the ridge regression

# Select the best alpha using the entire training set
alphas = np.arange(0.01,10,0.01)    
ridge_cv = RidgeCV(alphas=alphas)
cv_score = ridge_cv.fit(x_train, y_train)
cv_score.alpha_
# 0.01

ridge = Ridge(alpha=0.01)
ridge.fit(x_train_2,y_train_2)
pred = ridge.predict(x_cv)                      # Uses the validation set for prediction
mse = np.mean((pred - y_cv.values)**2)          # Calculates the mean squared error
mse  
# 0.046682854551005175

ridge.fit(x_train,y_train)
df_ridge = pd.DataFrame(np.expm1(ridge.predict(x_test)))

predictions = pd.DataFrame()
predictions['user.id'] = all_data['user.id'].unique()[len(y_train):,]
predictions['age'] = round(df_ridge)
predictions.to_csv('blog_result_1008_v4.csv',index=False)

