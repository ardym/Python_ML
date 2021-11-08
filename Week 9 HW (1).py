#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:27:33 2021

@author: ardalanmahdavieh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_excel("assignment8.xlsx")
df.columns
df=df.drop(["University name"], axis=1)

#1.	Use K means to find clusters in the data set, 
#please also interpret the clusters formed. 
#There is no right and wrong answer as long as you can explain the clusters. 
#Provide your output, and a summary of your results and analysis below.

#Scaling data using min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_df=scaler.fit_transform(df)

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
wcv=[] ##withi cluster variation
silk_score=[]
for i in range(2,11):
    km=KMeans(n_clusters=i)
    km.fit(scaled_df)
    wcv.append(km.inertia_)
    silk_score.append(silhouette_score(scaled_df,km.labels_))

plt.plot( range(2,11),wcv )
plt.xlabel( "no of clusters")
plt.ylabel("with in cluster variation")

plt.plot( range(2,11),silk_score )
plt.xlabel( "no of clusters")
plt.ylabel("silk score")
plt.grid()

#k=3
km=KMeans(n_clusters=3)
km.fit(scaled_df) 

#visulize the cluster
df["label"]=km.labels_

#interpert it
#cluster 0
c0=df.loc[df["label"]==0].describe()

#cluster 1
c1=df.loc[df["label"]==1].describe()

#cluster 2
c2=df.loc[df["label"]==2].describe()

###pt2####
#ward method
from scipy.cluster.hierarchy import dendrogram,linkage
linked=linkage(scaled_df,method="ward") 
dendrogram(linked)
plt.show()

#single method
linked=linkage(scaled_df,method="single")
dendrogram(linked)
plt.show()

#complete method
linked=linkage(scaled_df,method="complete") 
dendrogram(linked)
plt.show()

#average method
linked=linkage(scaled_df,method="average") 
dendrogram(linked)
plt.show()