#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:08:53 2021

@author: ardalanmahdavieh
"""

import pandas as pd
import seaborn as sns


#Dataframe
df=pd.read_csv("titanic_data-3.csv")

sns.pairplot(df)

#Create dummy variables
df=pd.get_dummies(df,drop_first=True)

#dropping all rows with missing values
df=df.dropna()


#Treat Survived as your y variable, and the other variables as your x variables. 
y=df["Survived"]
x=df.drop(["Survived"], axis=1)


#1. Build a KNN model to predict whether a passenger survives or not.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

#standarize the x
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)


#knn modeling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

knn=KNeighborsClassifier()
knn.fit(x_train_scaled,y_train)

#predictions
y_pred=knn.predict(x_test_scaled)

#f1
f1=f1_score(y_test,y_pred)
print("f1 score is", f1)
#f1 score is 0.6486486486486486

#2. See if the model can be improved using grid search. 

#creating a pipline before using grid search
from sklearn.pipeline import Pipeline
pipe=Pipeline([("scaler",MinMaxScaler()),("knn",KNeighborsClassifier())])
pipe.fit(x_train,y_train)
y_pred_pipe=pipe.predict(x_test)
f1=f1_score(y_test,y_pred_pipe)
print("f1 score is", f1)
#f1 score is 0.6486486486486486

#Param grid for hypertuining
param_grid={ 'knn__n_neighbors':range(1,50) , 'knn__p':[1,2] }
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(pipe,param_grid,verbose=3,scoring="f1")
grid.fit(x_train,y_train)

#best param
grid.best_params_


#using the tuned parameters 
pipe=Pipeline([("scaler",MinMaxScaler()),("knn",KNeighborsClassifier(n_neighbors=3,p=2))])

pipe.fit(x_train,y_train)
y_pred_pipe=pipe.predict(x_test)
f1=f1_score(y_test,y_pred_pipe)
print("f1 score is", f1)
#f1 score is 0.6883116883116883

#3. What happens when K=1? Check the model performance on the training and testing data.

knn1=KNeighborsClassifier(n_neighbors=1)
knn1.fit(x_train_scaled,y_train)

#predictions
y_pred=knn1.predict(x_test_scaled)

#f1
f1=f1_score(y_test,y_pred)
print("f1 score for K=1 is", f1)

#f1 score for K=1 is 0.5822784810126582

#4. What happens when K=N? Check the model performance on the training and testing data. Here, N=No of data points in the training set.

#len(X_train) to find find N 
len(x_train)

N=499

knnN=KNeighborsClassifier(n_neighbors=N)
knnN.fit(x_train_scaled,y_train)

#predictions
y_pred=knnN.predict(x_test_scaled)

#f1
f1=f1_score(y_test,y_pred)
print("f1 score for K=N is", f1)
#f1 score for K=N is 0.0






