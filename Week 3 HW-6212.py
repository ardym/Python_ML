#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:23:12 2021

@author: ardalanmahdavieh
"""

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import numpy as np


#Part 1: Titanic
df = pd.read_csv("Titanic-1.csv")
#Claim 1
df["Survived"].replace({0: "No", 1: "Yes"}, inplace=True)
sns.histplot(df["Survived"],bins=20, color="red")

#Claim 2
sns.countplot(x="Survived" ,hue="Sex", data=df)

#Claim 3
sns.countplot(x="Survived" ,hue="Pclass", data=df)

#Claim 4
df['Over 40'] = np.where(df['Age']>= 41, True, False)
sns.countplot(x="Over 40", data=df)

#Claim 5
df['Paid More Than $100'] = np.where(df['Fare']>= 100.00001, True, False)
sns.countplot(x="Paid More Than $100", data=df)

#Claim 6
df.groupby('Sex').mean()["Fare"].plot(kind='bar')
plt.xlabel("Sex")
plt.ylabel("Average Fare Price $")
plt.show()

#Claim 7
df.groupby('Pclass').mean()["Age"].plot(kind='bar')
plt.xlabel("Pclass")
plt.ylabel("Average Age")
plt.show()

#Claim 8
df.groupby('Pclass').mean()["Fare"].plot(kind='bar')
plt.xlabel("Pclass")
plt.ylabel("Average Fare Price $")
plt.show()

######Part 2: Stocks
TSLA_df=yf.download("TSLA","2019-01-01","2021-02-14")
GME_df=yf.download("GME","2019-01-01","2021-02-14")
BABA_df=yf.download("BABA","2019-01-01","2021-02-14")
NFLX_df=yf.download("NFLX","2019-01-01","2021-02-14")

plt.figure(figsize=(10,10)) 
plt.plot(TSLA_df["Open"],label="Tesla Stock",color="red")
plt.plot(GME_df["Open"],label="Gamestop",color="green")
plt.plot(BABA_df["Open"],label="Alibaba",color="blue")
plt.plot(NFLX_df["Open"],label="Netflix",color="orange")
plt.legend(fontsize=15, loc="upper left")
plt.title("Stocks", fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Date",fontsize=20)
plt.ylabel("Price $",fontsize=20)
plt.show()

######Part 3: Weights
df_w=pd.read_csv("weights.csv")

#Normal Distribution
plt.hist(df_w["Weight"], bins=20,edgecolor="black",label="Male height",color="red")
plt.legend()
plt.xlabel("Height")
plt.ylabel("Frequency")
plt.show()

#mean
df_mean = df_w["Weight"].mean()
print(df_mean)
#median
df_median = df_w["Weight"].median()
print(df_median)
#STDV
df_w.std()

####Part 4: Pokemon Charts
#Boxplot:
df = pd.read_excel("pokemon_data.xlsx")
df = df.drop(columns=['#','Name', 'Type 1', 'Type 2','Legendary'])
df1=pd.melt(df,id_vars="Generation")
plt.figure(figsize=(10,6))
sns.boxplot(x="variable", y="value" ,hue="Generation", data=df1)
plt.legend(title='Generation',loc="upper right")
plt.xlabel("Variable")
plt.ylabel("Value")
plt.show()

#Bar Chart:
df = pd.read_excel("pokemon_data.xlsx")
df = df.drop(columns=['#','Name', 'Type 1', 'Type 2','Generation'])
df1=pd.melt(df,id_vars="Legendary")
plt.figure(figsize=(10,6))
sns.barplot(x="variable", y="value" ,hue="Legendary", data=df1)
plt.legend(title='Legendary',loc="upper right")
plt.xlabel("Variable")
plt.ylabel("Value")
plt.show()