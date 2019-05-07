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

df3 = df[['perc_R_50c']].copy()

df4 = df[['perc_R_20']].copy()

print(df3)
print(df4)

c = df3/50
d = df4/20

print(c)
print(d)

## comparing responders when payoff is 50 vs. payoff = 20
print(stats.wilcoxon(c['perc_R_50c'], d['perc_R_20']))
