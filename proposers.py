##PROPOSERS

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
del df_initial['time']
del df_initial['name']
del df_initial['ID']

indexNames = df_initial[((df_initial['perc_P_50c'] > 50) | (df_initial['perc_P_50c'] < 0))
                        & ((df_initial['perc_R_50c'] < 0) | (df_initial['perc_R_50c'] > 50)) 
                        & ((df_initial['perc_P_20'] < 0) | (df_initial['perc_P_20'] > 20))
                        & ((df_initial['perc_R_20'] < 0) | (df_initial['perc_R_20'] > 20))].index
df = df_initial.drop(indexNames)

df[['perc_P_50c','perc_R_50c']] = df[['perc_P_50c','perc_R_50c']]/50
df[['perc_P_20','perc_R_20']] = df[['perc_P_20','perc_R_20']]/20

print(df)

stat, p = stats.wilcoxon(df['perc_P_50c'], df['perc_P_20'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')
    
print(stats.wilcoxon(df['perc_P_50c'], df['perc_P_20']))

print("               ")
print("mean of responder for 50 cents is:")
print(df['perc_P_50c'].mean())
print("               ")
print("mean of responder for 20 euros is:")
print(df['perc_P_20'].mean())

print("               ")
print("median of responder for 50 cents is:")
print(df['perc_P_50c'].median())
print("               ")
print("median of responder for 20 euros is:")
print(df['perc_P_20'].median())

df1 = [df['perc_P_20'], df['perc_P_50c']]
snsa_boxplot = sns.boxplot(data=df1)

import plotly.offline
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

import pandas as pd
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import json

def truncate(n, decimals=4):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

med_50cP = df['perc_P_50c'].median()
mean_50cP = truncate(df['perc_P_50c'].mean())

med_20P = df['perc_P_20'].median()
mean_20P = truncate(df['perc_P_20'].mean())

p_value = truncate(p)

trace1 = go.Table(
    header=dict(values=['','Mean','Median','Wilcoxon Statistic','P-value'],
                line = dict(color='#8c564b'),
                fill = dict(color='#17becf'),
                align = ['left'] * 5),
    cells=dict(values=[['Proposer 50 cents', 'Proposer 20 euros'],
                       [mean_50cP,mean_20P],
                      [med_50cP,mean_20P], [stat],[p_value]],
               line = dict(color='#8c564b'),
               fill = dict(color='#EDFAFF'),
               align = ['left'] * 5))

layout = dict(width=700, height=500)
data = [trace1]
fig = dict(data=data, layout=layout)
py.iplot(fig, filename = 'styled_table_P')
