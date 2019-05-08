import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn 
import seaborn
import scipy.stats as stats
from pandas import DataFrame 
import math 
from scipy.stats import mannwhitneyu

df_initial = pd.read_excel('./thewhyprobs.xlsx', names=['time','name','ID','perc_P_50c','perc_R_50c',
                                            'perc_P_20','perc_R_20','region'])


del df_initial['time']
del df_initial['name']
del df_initial['ID']

## deleting values outside of possible ranges 

indexNames = df_initial[((df_initial['perc_P_50c'] > 50) | (df_initial['perc_P_50c'] < 0))
                        & ((df_initial['perc_R_50c'] < 0) | (df_initial['perc_R_50c'] > 50)) 
                        & ((df_initial['perc_P_20'] < 0) | (df_initial['perc_P_20'] > 20))
                        & ((df_initial['perc_R_20'] < 0) | (df_initial['perc_R_20'] > 20))].index

## returns which ID is invalid according to possible range
print(indexNames)

## deletes values outside range of the dataframe 

df = df_initial.drop(indexNames)


## setting up for dutch vs. non dutch comparison 

df_dutch = df[(df['region'] == 'Dutch')]


df_nondutch = df[(df['region'] != 'Dutch')]

print(df_dutch)
print(df_nondutch)

##counts how many inputs in each 

print(df_dutch.count())
print(df_nondutch.count())

print(df)

mapping = {'Dutch': 1, 'Non-Dutch': 0}
df1 = df.replace({'region': mapping})
print(df1)

from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

cols = (col for col in df1.columns if col not in ['region'])
data = df1[cols]
target = df1['region']

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

svc_model = LinearSVC(random_state=0)

pred = svc_model.fit(data_train, target_train).predict(data_test)

print("LinearSVC accuracy : ",accuracy_score(target_test, pred, normalize = True))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gnb = GaussianNB()

#train the algorithm on training data and predict using the testing data
pred = gnb.fit(data_train, target_train).predict(data_test)
print(pred.tolist())

print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred, normalize = True))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(data_train, target_train)

pred = neigh.predict(data_test)

print ("KNeighbors accuracy score : ",accuracy_score(target_test, pred))
