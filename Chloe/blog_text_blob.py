
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge,RidgeCV,LassoCV

os.chdir('/Users/chloe/Desktop/UVa/Courses/SYS6018/Exercises/Kaggle/Blog/Data')

raw_train = pd.read_csv('train.csv')
raw_test = pd.read_csv('test.csv')

raw_train.info()

y_train = raw_train[['user.id','age']].drop_duplicates()
y_train = y_train.drop(['user.id'],axis=1)
y_train = y_train.reset_index(drop=True)
x_train = raw_train.drop(['age'],axis=1)
x_test = raw_test
all_data = pd.concat([x_train,x_test],ignore_index=True)
all_data.columns
all_data = all_data.drop(['post.id','date'],axis=1)

all_data_unique = all_data.iloc[:,0:4].drop_duplicates()
all_data_unique = all_data_unique.reset_index(drop=True)

# Create a variable to store the sentiment score for each user's comments
all_data_unique['senti'] = 0   
i = 0

# Calculates the sentiment score for each user
for id in all_data_unique['user.id']:
    texts = all_data[all_data['user.id'] == id][['text']]
    print (i)
    values = texts.values.tolist()
    sen = ''.join(str(value) for value in values)
    blob = TextBlob(sen)
    all_data_unique.iloc[i,4] = blob.sentiment.polarity
    i += 1

all_data_unique = all_data_unique.drop(['user.id'],axis=1)

all_dummies = pd.get_dummies(all_data_unique)

all_dummies.shape
# (19320, 55)

x_train = all_dummies.iloc[0:len(y_train),]
x_test = all_dummies.iloc[len(y_train):,]

x_train.shape
# (12880, 55)

x_test.shape
# (6440, 55)

y_train.shape
y_train.to_csv('y_train.csv',index=False)

y_train.describe()

sns.distplot(y_train)
fig = plt.figure()
stats.probplot(y_train, plot=plt)

# Normalize the "age" variable
y_train = np.log(y_train)
sns.distplot(y_train)
fig = plt.figure()
stats.probplot(y_train, plot=plt)

x_train_2, x_cv, y_train_2, y_cv = train_test_split(x_train, y_train,test_size=0.33, random_state=42)

# 1. Ridge Regression
# Select the best alpha using the entire training set
alphas = np.arange(0.01,10,0.01)    
ridge_cv = RidgeCV(alphas=alphas)
cv_score = ridge_cv.fit(x_train, y_train)
cv_score.alpha_
# 9.99

ridge = Ridge(alpha=9.99)
ridge.fit(x_train_2,y_train_2)
pred = ridge.predict(x_cv)                      # Uses the validation set for prediction
mse = np.mean((pred - y_cv.values)**2)          # Calculates the mean squared error
mse  
# 0.07941311707679259

# 2. Lasso Regression
# Select the best alpha using the entire training set
lasso_cv = LassoCV(alphas=alphas)
cv_score = lasso_cv.fit(x_train, y_train)
cv_score.alpha_
# 0.01

# Train lasso model with the training dataset after cross validation
lasso = Lasso(alpha=0.01, random_state=1)
lasso.fit(x_train_2,y_train_2)
pred = lasso.predict(x_cv)                      # Uses the validation set for prediction
mse = np.mean((pred - y_cv.values)**2)          # Calculates the mean squared error
mse
# 0.1263003273410172

# Train the models based on entire training dataset
ridge.fit(x_train,y_train)
lasso.fit(x_train,y_train)

# Predict on test dataset
# Since the target variable used in training has been log transformed,
# therefore we should calculate the result with exponential function.
df_ridge = pd.DataFrame(np.expm1(ridge.predict(x_test)))
df_lasso = pd.DataFrame(np.expm1(lasso.predict(x_test)))

# Since the performance of ridge regression is better than lasso,
# therefore we weight the result from ridge regression more.
result = 0.7*df_ridge + 0.3*df_lasso  

# Export the final result to csv file.
predictions = pd.DataFrame()
predictions['user.id'] = all_data['user.id'].unique()[len(y_train):,]
predictions['age'] = result
predictions.to_csv('blog_result_0924.csv',index=False)

