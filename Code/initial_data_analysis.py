##### DATA ANALYSIS AND CLEANING #####

# Import libraries and set working directory
import os
import pandas as pd
import numpy as np

os.chdir('J:\Data_Mining')

## Sources ##
# https://ww.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/
# https://www.nltk.org/book/ch03.html


# Import training set
train = pd.read_csv('train.csv')

## Simple data analysis ##
np.mean(train['age']) # average age is 23.54059386718018
train_length = len(train) # 442961 (number of blog posts)
train.groupby('gender').count() # 227840 females, 215121 males

# Topic and Sign Analysis
topics = train.groupby('topic').count()
len(topics) # 40 different topics
topics.sort_values(by=['age'], ascending=False) # indUnk and Student are the most popular topics
signs = train.groupby('sign').count()
signs.sort_values(by=['age'], ascending=False) # Taurus and Libra are the most popular
len(signs) # there are 12 different signs

# Age and gender
gender_age = train[['gender','age']] # subsetting gender and age to make analysis easier
females = gender_age[gender_age['gender'] == 'female']
males = gender_age[gender_age['gender'] == 'male']
np.mean(females['age']) # 23.523788623595507 (Basically the same as the average)
np.mean(males['age']) # 23.558392718516554 (Males are slightly older)
age_diff = np.mean(males['age']) - np.mean(females['age']) #  0.03460409492104688
np.min(train['age']) # 13 (minimum age)
np.max(train['age']) # 48 (maximum age)
train['age'].plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
    # Most are in mid twenties or really young (15 or younger)

# Investigating the length of the text entries
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
    # Older people have longer posts

def avg_word(sentence):
    words = sentence.split(' ')
    return (sum(len(word) for word in words)/len(words))

train['avg_word'] = train['text'].apply(lambda x: avg_word(x))
avg_char_per_word = np.mean(train['avg_word']) # 3.2439209616607747
greater_char_per_word = train[train['avg_word'] > avg_char_per_word]
np.mean(greater_char_per_word['age']) # 23.715549107240502
fewer_char_per_word = train[train['avg_word'] < avg_char_per_word]
np.mean(fewer_char_per_word['age']) # 23.256008254122612

# Older people use larger words

# Date analysis #
train['date'] = train['date'].str.replace(",","/")
train['date_split'] = train['date'].str.split('/')

train['day'] = [train['date_split'][x][0] for x in range(442961)]
train['month'] = [train['date_split'][x][1] for x in range(442961)]
train['year'] = [train['date_split'][x][2] for x in range(442961)]

train.groupby('year').count() # NOT ALL OF THE BLOGS ARE FROM 2004
"""
      word_count  char_count  avg_word  date_split     day   month  
year                                                                
              24          24        24          24      24      24  
1999          32          32        32          32      32      32  
2000         465         465       465         465     465     465  
2001        2769        2769      2769        2769    2769    2769  
2002       13256       13256     13256       13256   13256   13256  
2003       65345       65345     65345       65345   65345   65345  
2004      361054      361054    361054      361054  361054  361054  
2005           4           4         4           4       4       4  
2006          12          12        12          12      12      12 
"""
# Most of the posts are from 2004, but there are still some from mostly earlier
# years.

year_age = train[['year','age']]
year_age.groupby('year').mean()
"""
            age
year           
      24.000000
1999  26.906250
2000  25.520430
2001  28.257855
2002  26.116476
2003  23.543087
2004  23.406729
2005  20.250000
2006  18.166667
"""
# Ages get younger as years get more modern

month_age = train[['month','age']]
month_age_groupby = month_age.groupby('month').mean().sort_values(by='age')
len(np.unique(month_age['month'])) # 67 different months
    # There are many months that are not in English, which is the cause of 
    # 67 different months

# Investiagting English vs. Non-English entries
train['English'] = np.where((train['month'] == 'January') | (train['month'] == 'February') | (train['month'] == 'March') | (train['month'] == 'April') | (train['month'] == 'May') | (train['month'] == 'June') | (train['month'] == 'July') | (train['month'] == 'August') | (train['month'] == 'September') | (train['month'] == 'October') | (train['month'] == 'November') | (train['month'] == 'December') , 1, 0)
num_english_entries = train['English'].sum() # 438428
num_non_english_entries = train_length - num_english_entries # 4533
perc_english_posts = num_english_entries / train_length # 0.98976659344727869
    # The vast majority of posts (98.98%) are written in English
english_subset = train[train['English'] == 1]
english_subset['age'].mean() # 23.5513402428677

non_english_subset = train[train['English'] == 0]
non_english_subset['age'].mean() # 22.501213324509155 (slightly younger)

# Although English bloggers are over a year older than non-English bloggers,
# there are so few non-English blog posts that this variable proved not to be
# important in our regression analysis.

# Topic analysis
topic_age = train[['topic','age']] 
topic_age_mean = topic_age.groupby('topic').mean().sort_values(by='age')
"""
                               age
topic                             
Student                  17.908989
LawEnforcement-Security  21.457576
RealEstate               22.129785
Environment              22.144186
Sports-Recreation        22.321839
Maritime                 22.906832
Biotech                  23.045685
Chemicals                23.151493
Agriculture              23.610108
indUnk                   23.998319
Non-Profit               24.511399
Arts                     24.693826
Engineering              25.031237
Science                  25.070960
Architecture             25.633061
Tourism                  25.792541
Education                25.929914
Banking                  26.036380
Accounting               26.083295
Technology               26.361532
Communications-Media     26.418189
Marketing                26.634824
Military                 26.686908
HumanResources           26.732630
Law                      27.237913
Construction             27.838658
Manufacturing            27.843812
Fashion                  27.937205
InvestmentBanking        28.057912
Telecommunications       28.273133
Religion                 28.321119
Internet                 28.335477
BusinessServices         28.412276
Consulting               28.441630
Government               28.501014
Automotive               28.759259
Advertising              29.439648
Publishing               29.604560
Museums-Libraries        29.917299
Transportation           32.382334
"""
    # Students have an average age of around 18, while the oldest is Transportation at 32.382334
len(np.unique(train['topic'])) # 40 different topics

# User.id analysis
train_id = train[['user.id','age']]
id_groupby = train_id.groupby('user.id').count().sort_values(by='age', ascending=False)
id_groupby.head(15)
"""
          age
user.id      
17308    2301
8952     2261
4339     2237
3783     1951
13581    1843
17167    1731
13279    1542
3778     1533
5468     1533
5158     1505
18033    1351
3888     1337
5833     1304
8581     1279
18660    1265
"""
    # Some users account for at least 2000 blog entries in the dataset


### Overall Analysis notes ###
## Age ##
    # Average age = 23.54059386718018
    # Females are slightly younger than males (but probably doesn't matter)
    # Min age = 13, max age = 48
    # Histogram shows that the age range with the most blogs is between 23 and 27.
    # Skewed to the right, meaning that there are fewer older bloggers (ages 27-28)
    # compared to young bloggers (13-27)
## Gender ##
    # 227840 females, 215121 males (more females)
    # The age difference between female and male bloggers is not significant, as
    # male bloggers have an average age 0.035 years older compared to female bloggers
## Topics ##
    # 40 unique topics
    # Students are younger, transportation is oldest topic
## Sign ##
    # 12 different signs (but not a lot of different when it comes to age)
## Word Count and Character Count per post ##
    # Bloggers who use more words and characters are older 
    # Those who tweet longer words are older (but not by much)
## Dates ##
    # As the years get more current, the average age decreases 
## English ##
    # Non-English bloggers age is a whole year younger than the average English age,
    # however, there are far less foreign language blog entries compared to English
    # blog entries
    

# Conclusion #
# Although a lot of analysis was conducted, after experimenting with different 
# regressions, we found that the 'year' variable is the only one that is statistically
# significant and has a major impact on our results. You will notice in our
# regression python file that year is included in some of the regressions, improving
# our results.



