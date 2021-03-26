# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 22:06:20 2021

@author: Raouf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot
import seaborn as sns

#Reading the data
data=pd.read_csv("diabetes.csv")

#Finding null values
data.isnull().sum()
#Data types counts
data.dtypes.value_counts()
np.unique(data['Outcome'])

#Shape of the Dataset
data.shape

data.info()

data_c=data.copy()

data_c.describe()
pd.set_option('display.max_columns',10)
data.describe()

data_c.columns

#variable pregnancies
sns.distplot(data_c['Pregnancies'],kde=False)
sns.boxplot(y=data_c['Pregnancies'])

#variable Glucose
sns.distplot(data_c['Glucose'],kde=False)
sns.boxplot(y=data_c['Glucose'])

#variable BP
sns.distplot(data_c['BloodPressure'],kde=False)
sns.boxplot(y=data_c['BloodPressure'])
#Hence the data is clean

sns.regplot(x=data_c['Pregnancies'],y=data_c['Outcome'])
sns.distplot(data_c['Glucose'])

sns.pairplot(data_c,kind='scatter',hue='Outcome')

#Finding correlation
corr_matrix=data_c.corr()

sns.heatmap(corr_matrix,annot=True,cmap="RdYlGn")

"""RandomForest and KNN classifier model"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier




x=data_c.drop(['Outcome'],axis=1,inplace=False)
y=data_c['Outcome']

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=3)


#Baseline for model 
base_pred=np.mean(y_test)
base_pred=round(base_pred,3)
print(base_pred)

# ============================================================================= 
# Random Forest
# =============================================================================
rf=RandomForestClassifier(random_state=10)
model_rf=rf.fit(X_train,y_train)

#rf2=RandomForestRegressor()

model_predict=rf.predict(X_test)

rf_test1=model_rf.score(X_test,y_test)
rf_train1=model_rf.score(X_train,y_train)
print(rf_test1,rf_train1)


#Kmeans
from sklearn.cluster import KMeans
model=KMeans(n_clusters=3,n_jobs=4,random_state=3)
model.fit(x)
centers=model.cluster_centers_
print(centers)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
lgr=LogisticRegression()
model_lgr=lgr.fit(X_train,y_train)
model_predict_lgr=lgr.predict(X_test)

lgr_test=model_lgr.score(X_test,y_test)
lgr_train=model_lgr.score(X_train,y_train)
print(lgr_test,lgr_train)

#checking the probability of each patient
model_lgr.predict_proba(X_test)

