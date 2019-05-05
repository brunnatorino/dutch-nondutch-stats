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

df1 = df[['perc_P_50c']].copy()

df2 = df[['perc_P_20']].copy()

print(df1)
print(df2)

a = df1/50
b = df2/20

print(a)
print(b)

## comparing proposers when payoff is 50 vs. payoff is 20
print(stats.wilcoxon(a['perc_P_50c'], b['perc_P_20']))
