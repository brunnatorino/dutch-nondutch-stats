import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from pandas import DataFrame 
import math 
from scipy.stats import mannwhitneyu

df_initial = pd.read_excel('./thewhy_separate.xlsx', names = ['time','ID','region','culture',
                                                             'p_50','perc_p_50','r_50','perc_r_50','p_20','perc_p_20',
                                                             'r_20','perc_r_20'])
del df_initial['time']
del df_initial['ID']
del df_initial['p_50']
del df_initial['p_20']
del df_initial['r_50']
del df_initial['r_20']

print(df_initial.head())

mapping = {'Dutch': 1, 'Non-Dutch': 0}
df = df_initial.replace({'region': mapping})

mapping1 = {'Oceania':2,'Africa': 3, 'Asia': 4,'Eastern Europe':5,'Northern Europe':6, 'Western Europe':7}
df = df.replace({'culture': mapping1})

mappingmelt = {'perc_p_50':0,'perc_r_50': 1, 'perc_p_20': 2,'perc_r_20':3}
flights = df.replace({'variable': mappingmelt})

flights = pd.melt(df_initial, id_vars=['culture'], value_vars = ['perc_p_50','perc_r_50','perc_p_20','perc_r_20'])
mappingmelt = {'perc_p_50':0,'perc_r_50': 1, 'perc_p_20': 2,'perc_r_20':3}
flights = df_initial.replace({'variable': mappingmelt})
mapping = {'Dutch': 1, 'Non-Dutch': 0}
flights = flights.replace({'region': mapping})

print(flights)

ax = sns.heatmap(flights.corr(), annot = True)

mapping1 = {'Oceania':0,'Africa': 1, 'Asia': 2,'Eastern Europe':3,'Northern Europe':4, 'Western Europe':5}
flights = flights.replace({'culture': mapping1})

from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

cols = (col for col in flights.columns if col not in ['culture'])
data = flights[cols]
target = flights['culture']

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

neigh = KNeighborsClassifier(n_neighbors=4)

neigh.fit(data_train, target_train)

pred = neigh.predict(data_test)

print ("KNeighbors accuracy score : ",accuracy_score(target_test, pred))

## example, returns eastern europe
print(neigh.predict([[0,0.5,0.5,0.5,0.5]]))
