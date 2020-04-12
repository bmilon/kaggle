# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:44:46 2020
"""



import pandas as pd  
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from datetime import datetime
import pylab 
import statsmodels.api as sm
%matplotlib qt5

sns.set()

filename = r'C:\Users\mbhattac\Downloads\tab_files\avocado.csv'

data = pd.read_csv(filename)

data.Date = data.Date.map(lambda x : datetime.strptime(x, '%Y-%m-%d') )
    
# display prices for different types of avacodos 
for datatype in data.type.unique():
    print(datatype)
    sns.distplot(data.loc[data.type == datatype].AveragePrice, label=datatype)

plt.legend()



# display price by region 
height = int(data.region.unique().shape[0]/6)
width = 6 
f, axes = plt.subplots(height, width, figsize=(14, 14), sharex=True)

counterh=0
counterw=0
for datatype in data.region.unique():
    print(datatype)
    sns.distplot(data.loc[data.region == datatype].AveragePrice, label=datatype,ax=axes[counterh,counterw])
    plt.legend()
    counterh+=1
    if  counterh % height == 0:
        counterh=0
        counterw+=1
        

#  how does sales volume vary with price 
sns.lineplot(x=data.AveragePrice, y=data['Total Volume'])

# variation with time
for yeartype in data.year.unique():    
    sns.lineplot(y=data.loc[data.year == yeartype].AveragePrice, x=data.loc[data.year == yeartype]['Date'],markers=True,label=yeartype)
plt.legend()


# plot volatiity for each month in each year 
data['extractedmonthyear']= data.Date.map(lambda x: str(x.year) + '-' + str(x.month))
data['month']= data.Date.map(lambda x: str(x.month))
temp = data.groupby(['extractedmonthyear']).AveragePrice.std()
foo = pd.DataFrame()
foo['Variations']= temp.values
foo['Month'] = temp.index 
temp=foo
temp  = temp.sort_values(by=['Month'])
temp['Year']= temp.Month.map(lambda x : x.split('-')[0])
temp['MonthofYear']= temp.Month.map(lambda x : x.split('-')[1])



for year in temp.Year.unique():
    plt.figure()
    sns.scatterplot(x='MonthofYear',y='Variations', hue='Variations' , size='Variations' ,data=temp.loc[temp.Year ==  year ].sort_values(by=['MonthofYear']),)


data = data.set_index('Date')
# modeling trens in prices 

# pie chart 

fig, ax = plt.subplots()
ax.axis('equal')

tempdataforpiechart=data.groupby(['year']).sum()
yearpie, _ = ax.pie(tempdataforpiechart['Total Volume'], radius=0.5, labels=tempdataforpiechart.index.values)
plt.setp( yearpie, width=0.3, edgecolor='white')
tempdataforpiechart=data.groupby(['month']).sum()
monthpie, _ = ax.pie(tempdataforpiechart['Total Volume'], radius=2, labeldistance=1, labels=tempdataforpiechart.index.values)
plt.setp( monthpie, width=0.5, edgecolor='white')
plt.margins(0,0)
plt.show()


