# -*- coding: utf-8 -*-

import os
import pandas as pd
from textblob import TextBlob

# Read in raw datasets
# The raw data is too large to be uploaded to gitHub
raw_train = pd.read_csv('train.csv')
raw_test = pd.read_csv('test.csv')

# Get age of distinct user.ids
y_train = raw_train[['user.id','age']].drop_duplicates()
y_train = y_train.drop(['user.id'],axis=1)
y_train = y_train.reset_index(drop=True)

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

# The loop takes about 30 mins to finish
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

# Split the cleaned data to training set and testing set
x_train = all_dummies.iloc[0:len(y_train),]
x_test = all_dummies.iloc[len(y_train):,]

# Export the data to csv files
# x_train.to_csv('x_train.csv',index=False)
# x_test.to_csv('x_test.csv',index=False)
# y_train.to_csv('y_train.csv',index=False)


