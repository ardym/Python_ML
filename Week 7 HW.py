#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:16:23 2021

@author: ardalanmahdavieh
"""

import pandas as pd
import numpy as np
import seaborn as sns


#Dataframe
df=pd.read_csv("titanic_data-2-1.csv")

#drop id column
df=df.drop(["PassengerId"], axis=1)

#null values
sns.heatmap(df.isnull())

#dropping all rows with missing values
df=df.dropna()

#Create dummy variables
df=pd.get_dummies(df,drop_first=True)

#Treat Survived as your y variable, and the other variables as your x variables. 
y=df["Survived"]
x=df.drop(["Survived"], axis=1)

#1. Classification report showing precision, recall, F-score etc. 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=41)

#Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

#training set
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
y_pred_train=dt.predict(x_train)

print("Decision Tree training f1 score is:", f1_score(y_train,y_pred_train))
print("Decision Tree training recall score is:", recall_score(y_train,y_pred_train))
print("Decision Tree training precision score is:", precision_score(y_train,y_pred_train))
print("Decision Tree training accuracy score is:", accuracy_score(y_train,y_pred_train))

#Decision Tree training f1 score is: 1.0
#Decision Tree training recall score is: 1.0
#Decision Tree training precision score is: 1.0
#Decision Tree training accuracy score is: 1.0

#testing set
y_pred_test=dt.predict(x_test)
print("Decision Tree testing f1 score is:", f1_score(y_test,y_pred_test))
print("Decision Tree testing recall score is:", recall_score(y_test,y_pred_test))
print("Decision Tree testing precision score is:", precision_score(y_test,y_pred_test))
print("Decision Tree testing accuracy score is:", accuracy_score(y_test,y_pred_test))

#Decision Tree testing f1 score is: 0.8674698795180722
#Decision Tree testing recall score is: 0.8780487804878049
#Decision Tree testing precision score is: 0.8571428571428571
#Decision Tree testing accuracy score is: 0.8

#plotting the tree
from sklearn.tree import plot_tree
plt.figure(figsize=(10,10))
plot_tree(dt)
plt.show()

#Random Forest Model

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=500)

#train
rfc.fit(x_train,y_train)

y_pred_train=rfc.predict(x_train)
print("Random Forest training f1 score is:", f1_score(y_train,y_pred_train))
print("Random Forest training recall score is:", recall_score(y_train,y_pred_train))
print("Random Forest training precision score is:", precision_score(y_train,y_pred_train))
print("Random Forest training accuracy score is:", accuracy_score(y_train,y_pred_train))

#Random Forest training f1 score is: 1.0
#Random Forest training recall score is: 1.0
#Random Forest training precision score is: 1.0
#Random Forest training accuracy score is: 1.0

#testing set
y_pred_test=rfc.predict(x_test)
print("Random Forest testing f1 score is:", f1_score(y_test,y_pred_test))
print("Random Forest testing recall score is:", recall_score(y_test,y_pred_test))
print("Random Forest testing precision score is:", precision_score(y_test,y_pred_test))
print("Random Forest testing accuracy score is:", accuracy_score(y_test,y_pred_test))

#Random Forest testing f1 score is: 0.8604651162790697
#Random Forest testing recall score is: 0.9024390243902439
#Random Forest testing precision score is: 0.8222222222222222
#Random Forest testing accuracy score is: 0.7818181818181819

#2. Which model works better? Decision tree or random forest?

#both are having overffiting issues and simillar f1 score. RF has a better recall score and accuracy while DT has a better precision score



#3. Tune the hyper-parameters of the decision tree and random forest model. How the performance of the tuned models compare with un-tuned models?

#DT Optimization
#max depth of tree
dt.tree_.max_depth

#using grid search to imporve the tree
parameter_grid={"max_depth":range(2,8),"min_samples_split":range(2,6)}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(dt,parameter_grid,verbose=3,scoring="f1")

grid.fit(x_train,y_train)
grid.best_params_

#{'max_depth': 3, 'min_samples_split': 4}

#use optimized values to improve the model

dt=DecisionTreeClassifier(max_depth=3,min_samples_split=4)
dt.fit(x_train,y_train)

#evalute our DT 

#training set
y_pred_train=dt.predict(x_train)
print("Tuned Decision Tree training f1 score is:", f1_score(y_train,y_pred_train))
print("Tuned Decision Tree training recall score is:", recall_score(y_train,y_pred_train))
print("Tuned Decision Tree training precision score is:", precision_score(y_train,y_pred_train))
print("Tuned Decision Tree training accuracy score is:", accuracy_score(y_train,y_pred_train))

#Tuned Decision Tree training f1 score is: 0.8823529411764706
#Tuned Decision Tree training recall score is: 0.9146341463414634
#Tuned Decision Tree training precision score is: 0.8522727272727273
#Tuned Decision Tree training accuracy score is: 0.84375

#testing set
y_pred_test=dt.predict(x_test)
print("Tuned Decision Tree testing f1 score is:", f1_score(y_test,y_pred_test))
print("Tuned Decision Tree testing recall score is:", recall_score(y_test,y_pred_test))
print("Tuned Decision Tree testing precision score is:", precision_score(y_test,y_pred_test))
print("Tuned Decision Tree testing accuracy score is:", accuracy_score(y_test,y_pred_test))

#Tuned Decision Tree testing f1 score is: 0.8735632183908046
#Tuned Decision Tree testing recall score is: 0.926829268292683
#Tuned Decision Tree testing precision score is: 0.8260869565217391
#Tuned Decision Tree testing accuracy score is: 0.8

#RFC Optimization
parameter_grid={"max_depth":range(2,16),"min_samples_split":range(2,6)}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(rfc,parameter_grid,verbose=3,scoring="f1")

grid.fit(x_train,y_train)
grid.best_params_
#{'max_depth': 14, 'min_samples_split': 2}

#using optimized values for RFC
rfc=RandomForestClassifier(n_estimators=500,max_depth=14,min_samples_split=2)

#train
rfc.fit(x_train,y_train)

y_pred_train=rfc.predict(x_train)
print("Tuned Random Forest training f1 score is:", f1_score(y_train,y_pred_train))
print("Tuned Random Forest training recall score is:", recall_score(y_train,y_pred_train))
print("Tuned Random Forest training precision score is:", precision_score(y_train,y_pred_train))
print("Tuned Random Forest training accuracy score is:", accuracy_score(y_train,y_pred_train))

#Tuned Random Forest training f1 score is: 0.993939393939394
#Tuned Random Forest training recall score is: 1.0
#Tuned Random Forest training precision score is: 0.9879518072289156
#Tuned Random Forest training accuracy score is: 0.9921875

#testing set
y_pred_test=rfc.predict(x_test)
print("Tuned Random Forest testing f1 score is:", f1_score(y_test,y_pred_test))
print("Tuned Random Forest testing recall score is:", recall_score(y_test,y_pred_test))
print("Tuned Random Forest testing precision score is:", precision_score(y_test,y_pred_test))
print("Tuned Random Forest testing accuracy score is:", accuracy_score(y_test,y_pred_test))

#Tuned Random Forest testing f1 score is: 0.8604651162790697
#Tuned Random Forest testing recall score is: 0.9024390243902439
#Tuned Random Forest testing precision score is: 0.8222222222222222
#Tuned Random Forest testing accuracy score is: 0.7818181818181819

#6. Create an ensemble using the classification models learned so far. See if the ensemble works better than the individual models.

from sklearn.ensemble import VotingClassifier

vclf= VotingClassifier(estimators=[('DT',dt,),('RF',rfc)])

vclf.fit(x_train,y_train)

#train model
y_pred_train=vclf.predict(x_train)

print("Voting Classifier training f1 score is:", f1_score(y_train,y_pred_train))
print("Voting Classifier training recall score is:", recall_score(y_train,y_pred_train))
print("Voting Classifier training precision score is:", precision_score(y_train,y_pred_train))
print("Voting Classifier training accuracy score is:", accuracy_score(y_train,y_pred_train))

#Voting Classifier training f1 score is: 0.9554140127388536
#Voting Classifier training recall score is: 0.9146341463414634
#Voting Classifier training precision score is: 1.0
#Voting Classifier training accuracy score is: 0.9453125

#test model

y_pred_test=vclf.predict(x_test)
print("Voting Classifier testing f1 score is:", f1_score(y_test,y_pred_test))
print("Voting Classifier testing recall score is:", recall_score(y_test,y_pred_test))
print("Voting Classifier testing precision score is:", precision_score(y_test,y_pred_test))
print("Voting Classifier testing accuracy score is:", accuracy_score(y_test,y_pred_test))

#Voting Classifier testing f1 score is: 0.8604651162790697
#Voting Classifier testing recall score is: 0.9024390243902439
#Voting Classifier testing precision score is: 0.8222222222222222
#Voting Classifier testing accuracy score is: 0.7818181818181819