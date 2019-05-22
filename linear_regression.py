## RESPONDERS 

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

print(df.head())

df[['perc_P_50c','perc_R_50c']] = df[['perc_P_50c','perc_R_50c']]/50
df[['perc_P_20','perc_R_20']] = df[['perc_P_20','perc_R_20']]/20

print(df.head())

## histograms with plotly, skip for lin. reg.

x0 = df['perc_P_20']
x1 = df['perc_P_50c']

trace1 = go.Histogram(
    x=x0,
    histnorm='percent',
    name='20 euros',
    xbins=dict(
        start=0,
        end=1,
        size=0.1
    ),
    marker=dict(
        color='#FFD7E9',
    ),
    opacity=0.75
)
trace2 = go.Histogram(
    x=x1,
    name='50 cents',
    xbins=dict(
        start=0,
        end=1,
        size=0.1
    ),
    marker=dict(
        color='#EB89B5'
    ),
    opacity=0.75
)
data = [trace1, trace2]

layout = go.Layout(
    title='Sampled Results',
    xaxis=dict(
        title='Value'
    ),
    yaxis=dict(
        title='Count'
    ),
    bargap=0.2,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='styled histogram')


mapping = {'Dutch': 1, 'Non-Dutch': 0}
df1 = df.replace({'region': mapping})
print(df1.head())


print("Shape:", df1.shape)
print("\nFeatures:", df1.columns)

X = df1[df1.columns[:-1]] ## all columns but the region

y = df1[df1.columns[-1]] ## all columns but the index 

target_names = df1[df1.columns[-1]]
print(target_names.head())

print("\nFeature matrix:\n", X.head()) 
print("\nResponse vector:\n", y.head())

print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 

# printing the shapes of the new X objects 
print(X_train.shape) 
print(X_test.shape) 
  
# printing the shapes of the new y objects 
print(y_train.shape) 
print(y_test.shape)

import sklearn 
from sklearn import datasets, linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

p1 = go.Scatter(x=X_test['perc_R_50c'],
                y=y_test, # Assuming y_test is a numpy array or pandas series
                          # if it is also a dataframe you have to specify the column
                mode='markers',
                marker=dict(color='black')
               )

p2 = go.Scatter(x=X_test['perc_R_20'],
                y=regr.predict(X_test),
                mode='lines',
                line=dict(color='blue', width=3)
                )
layout = go.Layout(xaxis=dict(ticks='', showticklabels=False,
                              zeroline=False),
                   yaxis=dict(ticks='', showticklabels=False,
                              zeroline=False),
                   showlegend=False, hovermode='closest')

fig = go.Figure(data=[p1, p2], layout=layout)

py.offline.iplot(fig)
