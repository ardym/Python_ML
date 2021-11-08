"""Assignment: Use the alcohol data set  provided here as alcohol.xlsx
Please write your code after the comments, and upload the completed .py file along with updated excel and csv files"""

#Import the data and relevant libraries

import pandas as pd
df=pd.read_excel("alcohol.xlsx")

#sort data based on decreasing beer_servings and assign it to a variable called sort_beer

sort_beer=df.sort_values(by=["beer_servings"],ascending=False)


#which country drinks the highest spirit_servings

sort_spirit=df.sort_values(by=["spirit_servings"],ascending=False)
#Answer: Grenada

#get all rows with beer servings greater than 100

df.loc[df["beer_servings"]>100]

#get all rows with beer servings greater than 100 in Asia (AS)

df.loc[(df["continent"]=="AS") & (df["beer_servings"]>100)]

#get all the rows for continent "A"

df.loc[df["continent"]=="A"]

#get the mean alcohol consumption per continent for every column (hint: use groupby)

grouped_df = df.groupby("continent")
mean_df = grouped_df.mean()
mean_df = mean_df.reset_index()
print(mean_df)

#get the median alcohol consumption per continent for every column

grouped_df = df.groupby("continent")
median_df = grouped_df.median()
median_df = median_df.reset_index()
print(median_df)

#Create a new column called total_servings which is the sum of beer_servings, spirit_servings, wine_servings

df['total_servings'] = df.sum(axis=1)

#Sort the data based on total_servings and state which country drinks most and which drinks least

sort_spirit=df.sort_values(by=["total_servings"],ascending=False)
#Answer: Andorra

#Read column beer_servings

df['beer_servings']

#Read columns beer_servings and wine_servings

df[['beer_servings', 'wine_servings']]

#for countries that drink more than 200 servings of beer, change their (country names) names to "beer nation"

df.loc[df.beer_servings > 200, 'country'] = 'beer nation'
    
#save the data frame as an Excel file with name updated_drinks_excel

df.to_excel("updated_drinks_excel.xlsx")

#save the data frame as a csv file with name updated_drinks_csv

df.to_csv("updated_drinks_csv.csv")

#Write a program to print the cube of numbers from 2 to 100 (including both 2 and 100)

Cubes=[]
for i in range(2,101):
    Cubes.append(i**3)
print(Cubes)

#Write a program to print the cube of even numbers from 2 to 100 (including both 2 and 100)

L = list(range(2, 101))
EvenList = [x for x in L if x % 2 == 0]

Cubes=[]
for i in EvenList:
    Cubes.append(i**3)
print(Cubes)
        
        
#Give 5 examples of reserved words in python

#and
#del
#for
#is
#raise

#give 4 examples of bad variable names and state why they are invalid

#Variables in python must start with a letter or underscore and they must consist of letters, numbers and underscores. Therefore the following are examples of "bad variables":

 #7numbers     
 #$sign  
 #var.15
 #(df)