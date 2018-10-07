# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Import libraries and set working directory
import os
import pandas as pd
import numpy as np
import nltk, re, pprint
from datetime import datetime as DT
from nltk import word_tokenize
nltk.download('punkt')
from nltk.corpus import gutenberg
from nltk.corpus import stopwords
from time import strptime

os.chdir('J:\Data_Mining')

## Sources ##
# https://ww.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/
# https://www.nltk.org/book/ch03.html


# Import training set
train = pd.read_csv('train.csv')

## Simple data analysis ##
np.mean(train['age']) # average age is 23.54059386718018
len(train) # 442961 (number of blog posts)
train.groupby('gender').count() # 227840 females, 215121 males

topics = train.groupby('topic').count()
len(topics) # 40 different topics
topics.sort_values(by=['age'], ascending=False) # indUnk and Student are the most popular topics
signs = train.groupby('sign').count()
signs.sort_values(by=['age'], ascending=False) # Taurus and Libra are the most popular
len(signs)

gender_age = train[['gender','age']]

# Age and gender
females = gender_age[gender_age['gender'] == 'female']
males = gender_age[gender_age['gender'] == 'male']
np.mean(females['age']) # 23.523788623595507 (Basically the same as the average)
np.mean(males['age']) # 23.558392718516554 (Males are slightly older)
np.min(train['age']) # 13 (minimum age)
np.max(train['age']) # 48 (maximum age)


## Adding analysis to the training dataset ##
train['word_count'] = train['text'].apply(lambda x: len(str(x).split(" ")))
train['char_count'] = train['text'].str.len()

avg_word_count = np.mean(train['word_count']) # 237.52244328507476 (average post)
avg_char_count = np.mean(train['char_count']) # 1142.8452888629022

greater_avg_word = train[train['word_count'] > avg_word_count]
greater_age = np.mean(greater_avg_word['age']) # 24.119490798667247 (older than average age)
fewer_avg_word = train[train['word_count'] < avg_word_count]
fewer_age = np.mean(fewer_avg_word['age']) # 23.266715220203594 (younger than average age)

greater_char = train[train['char_count'] > avg_char_count]
greater_char_age = np.mean(greater_char['age']) # 24.153848328104242
fewer_char = train[train['char_count'] < avg_char_count]
fewer_char_age = np.mean(fewer_char['age']) # 23.252696180066014

def avg_word(sentence):
    words = sentence.split(' ')
    return (sum(len(word) for word in words)/len(words))

train['avg_word'] = train['text'].apply(lambda x: avg_word(x))
avg_char_per_word = np.mean(train['avg_word']) # 3.2439209616607747
greater_char_per_word = train[train['avg_word'] > avg_char_per_word]
np.mean(greater_char_per_word['age']) # 23.715549107240502
fewer_char_per_word = train[train['avg_word'] < avg_char_per_word]
np.mean(fewer_char_per_word['age']) # 23.256008254122612

# Date analysis #
train['date'] = train['date'].str.replace(",","/")
train['date_split'] = train['date'].str.split('/')

train['day'] = [train['date_split'][x][0] for x in range(442961)]
train['month'] = [train['date_split'][x][1] for x in range(442961)]
train['year'] = [train['date_split'][x][2] for x in range(442961)]

train.groupby('year').count() # NOT ALL OF THE BLOGS ARE FROM 2004

year_age = train[['year','age']]
year_age.groupby('year').mean()  # Ages get younger as years get more modern

month_age = train[['month','age']]
month_age_groupby = month_age.groupby('month').mean().sort_values(by='age')
    # MOST DATES ARE NOT IN ENGLISH SO THIS IS HARD TO DECIPHER
len(np.unique(month_age['month'])) # 67 different months

topic_age = train[['topic','age']] 
topic_age_mean = topic_age.groupby('topic').mean().sort_values(by='age')
    # Students have an average age of around 18, while the oldest is Transportation at 32.382334
    

# Overall Analysis notes #
## Age ##
    # Average age = 23.54059386718018
    # Females are slightly younger than males
    # Min age = 13, max age = 48
## Gender ##
    # 227840 females, 215121 males (more females)
## Topics ##
    # 40 unique topics
    # Students are younger, transportation is oldest topic
## Sign ##
    # 12 different signs
## Word Count and Character Count per post ##
    # Bloggers who use more words and characters are older 
    # Those who tweet longer words are older (but not by much)
## Dates ##
    # As the years get more current, the average age decreases 
    # hard to decipher months



# Text analytics (very unsuccessful) #
train['text'][15] # look at one text entry

sample = train.iloc[0:100,] # takes the first 100 rows as a sample
sample['text'][15]
type(sample) # already a dataframe

texts = pd.DataFrame(sample['text']) # just takes the text columns
texts['text'][15]

tokenized_sample = sample['text'].apply(word_tokenize)
len(tokenized_sample) # 100 samples

text_sample = nltk.Text(tokenized_sample)
len(text_sample) # still 100

type(tokenized_sample)
token_df = pd.DataFrame(tokenized_sample)

