#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:45:54 2021

@author: ardalanmahdavieh
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Dataframe
df=pd.read_csv("titanic_data-2.csv")


#Split by first and last name
df[['Last','First']] = df.Name.str.split(",",expand=True,)

#drop the following variables since we cant convert string to float in this case
df=df.drop(["Cabin", "Ticket", "Name", "First"], axis=1)

#Create dummy variables for Embarked and Sex
df=pd.get_dummies(df,drop_first=True)

#dropping all rows with missing values
df=df.dropna()

#Treat Survived as your y variable, and the other variables as your x variables. 
y=df["Survived"]
x=df.drop(["Survived"], axis=1)

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

#Logistic Regression Model
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(solver="liblinear")
logmodel.fit(x_train,y_train)

#make prediction
y_pred=logmodel.predict(x_test)

#to get probability
y_probab=logmodel.predict_proba(x_test)

#Precision, recall, F-score, and confusion matrix:
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score, f1_score

C_mat=pd.DataFrame(confusion_matrix(y_test,y_pred,labels=[0,1]), index=["Actual:0","Actual:1"],columns=["Pred:0","Pred:1"])
print(C_mat)

print("Accuracy score is", accuracy_score(y_test,y_pred))
print("Recall score is", recall_score(y_test,y_pred))
print("Precision score is", precision_score(y_test,y_pred))
print("f1 score is",f1_score(y_test,y_pred))

#Accuracy score is 0.794392523364486
#Recall score is 0.7093023255813954
#Precision score is 0.7625
#f1 score is 0.7349397590361445

#SVM
#build the model
from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train) 

#prediction
y_pred=model.predict(x_test)

#evaluate the model
from sklearn.metrics import f1_score,recall_score
print("f1 score is",f1_score(y_test,y_pred))
print("Recall score is", recall_score(y_test,y_pred))

#grid search to find the best value of C, gamma, kernel
param_grid={"C":[1,10,100],"gamma":[1,0.1,0.01],"kernel":["rbf","linear"]}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(model,param_grid,verbose=3,scoring="f1")
grid.fit(x_train,y_train)
grid.best_params_

#With the tuned parameters
model=SVC(C=10,gamma=1,kernel="linear")
model.fit(x_train,y_train) 

#prediction
y_pred=model.predict(x_test)

#evaluate the model

C_mat=pd.DataFrame(confusion_matrix(y_test,y_pred,labels=[0,1]), index=["Actual:0","Actual:1"],columns=["Pred:0","Pred:1"])
print(C_mat)

print("Accuracy score is", accuracy_score(y_test,y_pred))
print("Recall score is", recall_score(y_test,y_pred))
print("Precision score is", precision_score(y_test,y_pred))
print("f1 score is",f1_score(y_test,y_pred))

#Linear Regression

df=pd.read_csv("titanic_data-2.csv")

#Split by first and last name
df[['Last','First']] = df.Name.str.split(",",expand=True,)

#drop the following variables since we cant convert string to float in this case
df=df.drop(["Cabin", "Ticket", "Name", "First"], axis=1)

#Create dummy variables for Embarked and Sex
df=pd.get_dummies(df,drop_first=True)

#dropping all rows with missing values
df=df.dropna()

#Treat Survived as your y variable, and the other variables as your x variables. 
y=df["Survived"]
x=df.drop(["Survived"], axis=1)

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

#build the model
from sklearn.linear_model import LinearRegression
model=LinearRegression() 
model.fit(x_train,y_train) 

#prediction
y_pred=model.predict(x_test)

#Prediction includes negative values and values largers than one