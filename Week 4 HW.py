#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:05:59 2021

@author: ardalanmahdavieh
"""

import pandas as pd

df = pd.read_csv("USA_Housing.csv")
df.columns

#  Character strings, cant be used for regression model. Cant create dummy variables since they are all so different from each other.
df=df.drop(["Address"], axis=1) 

#split into x and y variables
x=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
y=df['Price']

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

#Regression
from sklearn.linear_model import LinearRegression
lm=LinearRegression() #not trained yet
lm.fit(x_train,y_train) #training the model

#The model:
lm.coef_
#([2.16398550e+01, 1.65729214e+05, 1.20958349e+05, 1.94909254e+03, 1.52262240e+01])
lm.intercept_
#-2645289.864342805

#Prediction(y-hat)
predictions=lm.predict(x_test)

#evaluting the model using R^2 and RMSE

from sklearn.metrics import r2_score,mean_squared_error

print("R2 is:", r2_score(y_test,predictions))

mse=mean_squared_error(y_test,predictions)
print("RMSE is :",mse**0.5)

#R2 is: 0.9166912271539773
#RMSE is : 102798.09614448404

#using a loop to find the optimal K value with the help of feature selection
i=1

while i<=5:
    from sklearn.feature_selection import SelectKBest,f_regression
    bestfeatures=SelectKBest(score_func=f_regression, k=i)

    new_x=bestfeatures.fit_transform(x,y)

#train test split

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(new_x,y,test_size=0.3,random_state=1)

#build the model
    from sklearn.linear_model import LinearRegression
    lm=LinearRegression() 
    lm.fit(x_train,y_train) 

#prediction
    y_pred=lm.predict(x_test)

#evaluate the model
    from sklearn.metrics import r2_score,mean_squared_error
    print("R2 is:", r2_score(y_test,y_pred))

    mse=mean_squared_error(y_test,y_pred)
    print("RMSE is :",mse**0.5)
    i=i+1

#k=4 is the best, 'Avg. Area Number of Bedrooms is dropped
#k1
#R2 is: 0.42939928773484226
#RMSE is : 269033.18772319076
#k2
#R2 is: 0.6293750896758311
#RMSE is : 216823.82702923586
#k3
#R2 is: 0.7978739418731635
#RMSE is : 160121.94470518373
#k4
#R2 is: 0.9166922874042369
#RMSE is : 102797.44199936118
#k5
#R2 is: 0.9166912271539759
#RMSE is : 102798.09614448491

