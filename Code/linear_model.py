
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import KFold

# Read in the cleaned data
train_temp = pd.read_csv('x_train.csv')
test_temp = pd.read_csv('x_test.csv')
Y_train = pd.read_csv('y_train.csv')
tfidf_train = pd.read_csv('tfidf80_train.csv')
tfidf_test = pd.read_csv('tfidf80_test.csv')

# Deal with the tfidf data
tfidf_train.rename(columns={'Unnamed: 0':'user.id'},inplace=True)
tfidf_test.rename(columns={'Unnamed: 0':'user.id'},inplace=True)

# The terms in testing set are different from those in training set
# In order to avoid being informed by the testing set when building the model,
# we only keep the terms in training set
cols = tfidf_train.columns     # Get the terms in training set
tfidf = tfidf_train.append(tfidf_test)  # Combine the tfidf of training set and testing set together
tfidf = tfidf[cols]            # Only keep terms in training set
tfidf = tfidf.fillna('0')      # Convert NA to zero


# Log transform the highly-skewed age data
Y_train = np.log(Y_train)  

# Merge the data with tfidf data 
data = pd.concat([train_temp,test_temp],axis=0)
tfidf.rename(columns={'Unnamed: 0':'user.id'},inplace=True)
complete_data = pd.merge(data,tfidf,on='user.id')

# Drop user.id
final_data = complete_data.drop(['user.id'],axis=1)

# Split the final dataset to training and testing set
X_train = final_data.iloc[0:len(Y_train),]
X_test = final_data.iloc[len(Y_train):,]

# Build the model

# 1. Ridge regression
# Select the best alpha based on entire dataset
alphas = np.arange(0.01,10,0.01)    
ridge_cv = RidgeCV(alphas=alphas)
cv_score = ridge_cv.fit(X_train, Y_train)
cv_score.alpha_
# 0.01

# 2. Lasso Regression
# Select the best alpha based on entire dataset
alphas = np.arange(0.01,10,0.01)    
lasso_cv = LassoCV(alphas=alphas)
cv_score = lasso_cv.fit(X_train, Y_train)
cv_score.alpha_
# 0.01

# Cross Validation
# We use the kFold package to automatically generate the index of validation
def kFoldValidation(k, model, X_train, Y_train):
    mseTotal = 0
    kf = KFold(n_splits=k, random_state=52, shuffle=True)
    for train_index, test_index in kf.split(X_train):     # Generate the index
        x_train, x_test = X_train.iloc[train_index,], X_train.iloc[test_index,]
        y_train, y_test = Y_train.iloc[train_index,], Y_train.iloc[test_index,]
        model.fit(x_train,y_train)
        pred = model.predict(x_test)                      # Use the validation set for prediction
        mse = np.mean((pred - y_test.values)**2)          # Calculate the mean squared error
        mseTotal += mse
    return mseTotal/5    # Return the average mean squared error

               
ridge = Ridge(alpha=0.01)
lasso = Lasso(alpha=0.01)
ridgeMse = kFoldValidation(5, ridge, X_train, Y_train)
ridgeMse    # 0.04608180089653157
lassoMse = kFoldValidation(5, lasso, X_train, Y_train)
lassoMse    # 0.12668755069232834

# Base on k-fold cross validation, ridge regression has a much lower
# average mean squared error. Therefore, we choose to use ridge regression
# in our final model
# Fit the model with entire datasets
ridge.fit(X_train, Y_train)

# Predict on the test set with ridge regression
df_ridge = pd.DataFrame(np.expm1(ridge.predict(X_test)))

# Exports the data to csv
predictions = pd.DataFrame()
predictions['user.id'] = complete_data['user.id'].unique()[len(Y_train):,]
predictions['age'] = round(df_ridge)
# predictions.to_csv('blog_result_1009_V2.csv',index=False)