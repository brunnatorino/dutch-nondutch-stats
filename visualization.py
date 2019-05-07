import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from pandas import DataFrame 
import math 
from scipy.stats import mannwhitneyu

df_initial = pd.read_excel('./thewhyprobs.xlsx', names=['time','name','ID','perc_P_50c','perc_R_50c',
                                            'perc_P_20','perc_R_20','region'])
print(df_initial)

##don't run this before running stats tests!!!!!!!!!!

del df_initial['time']
del df_initial['name']
del df_initial['ID']

# delete all rows for which column 'Age' has value greater than 30 and Country is India 
indexNames = df_initial[((df_initial['perc_P_50c'] > 50) | (df_initial['perc_P_50c'] < 0))
                        & ((df_initial['perc_R_50c'] < 0) | (df_initial['perc_R_50c'] > 50)) 
                        & ((df_initial['perc_P_20'] < 0) | (df_initial['perc_P_20'] > 20))
                        & ((df_initial['perc_R_20'] < 0) | (df_initial['perc_R_20'] > 20))].index
df = df_initial.drop(indexNames)
print(df)

df1=pd.melt(df, id_vars='region',value_vars=['perc_P_50c','perc_R_50c',
                                               'perc_P_20','perc_R_20'])
                                               
sns_boxplot = sns.boxplot(data=df)
sns_boxplot.figure.savefig("boxplot")

sns_swarm = sns.swarmplot(x='variable', y='value', data=df1)
sns_swarm.figure.savefig("swarm")

g = sns.catplot(x='variable', y='value', data=df1, height = 5, aspect =.9)
g.figure.savefig("catplot")

sns.set(style="ticks")
g = sns.catplot(x='region', y='perc_P_20', data=df, height = 5, aspect =.9)
g = sns.catplot(x='region', y='perc_P_50c', data=df, height = 5, aspect =.9)
g = sns.catplot(x='region', y='perc_R_20', data=df, height = 5, aspect =.9)
g = sns.catplot(x='region', y='perc_R_50c', data=df, height = 5, aspect =.9)

sns.distplot(a, hist=False, rug=True);  ##blue
sns.distplot(b, hist=False, rug=True);  ##orange

sns.distplot(c, hist=False, rug=True);##blue
sns.distplot(d, hist=False, rug=True);##orange

sns.boxplot(data=df)


************************ separate kernel

df1=pd.melt(df, id_vars='region',value_vars=['perc_P_50c','perc_R_50c',
                                               'perc_P_20','perc_R_20'])
sns.swarmplot(x='variable', y='value', data=df1)

del df['perc_P_20']
del df['perc_R_20']
del df['region']

print(df)

g = sns.catplot(x='variable', y='value', data=df1, height = 5, aspect =.9)


## linear regression

import seaborn as sn; sns.set(color_codes=True)

ax = sns.regplot(x='perc_P_50c',y='perc_R_50c', data=df, ci = 95)

ax = sns.regplot(x=b['perc_P_20'],y=d['perc_R_20'], data=df, ci = 95)
                                              
