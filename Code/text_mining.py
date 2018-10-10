
# -*- coding: utf-8 -*-

from nltk.corpus import stopwords
import pandas as pd
import string

# Read in raw datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Define a function to remove the punctuations and stop words in text
# We conduct white space removing, number removing and stemming in R
def clean(text):
    puncs = set(string.punctuation) 
    stopWords = set(stopwords.words('english'))
    temp = ' '.join([i for i in text.lower().split() if i not in stopWords])
    cleaned_text = ''.join(ch for ch in temp if ch not in puncs)
    return cleaned_text

# Create a dataset to store the aggregated training set
new_train = pd.DataFrame()
id_train = train['user.id'].unique()     # Get distinct user.id in training set 
new_train['user.id'] = id_train
new_train['texts'] = 0     

i = 0
for id in id_train:     
    # aggregate the texts of the same user.id
    texts = train[train['user.id'] == id][['text']] 
    values = texts.values.tolist()
    sen = ''.join(str(value) for value in values)
    new_train.iloc[i,1] = sen

# Create a dataset to store the aggregated testing set
id_test = test['user.id'].unique()
new_test = pd.DataFrame()
new_test['user.id'] = id_test
new_test['texts'] = 0
i = 0
for id in id_test:     
    # aggregate the texts of the same user.id
    texts = test[test['user.id'] == id][['text']] 
    values = texts.values.tolist()
    sen = ''.join(str(value) for value in values)
    new_test.iloc[i,1] = sen

# Export the aggregated dataset to csv file
# new_train.to_csv('x_train_aggregate.csv',index=False)
# new_test.to_csv('x_test_aggregate.csv',index=False)

# Create a new column to store the cleaned text
new_train['cleaned_text'] = 0
for i in range(new_train.shape[0]):
    text = new_train.iloc[i,1]
    new_train.iloc[i,2] = clean(text)

new_test['cleaned_text'] = 0
for i in range(new_test.shape[0]):
    text = new_test.iloc[i,1]
    new_test.iloc[i,2] = clean(text)

# Exported the cleaned text to csv files
x_train_for_tm = new_train[['user.id','cleaned_text']]
x_test_for_tm = new_test[['user.id','cleaned_text']]
x_train_for_tm.to_csv('x_train_for_tm.csv',index=False)
x_test_for_tm.to_csv('x_test_for_tm.csv',index=False)
