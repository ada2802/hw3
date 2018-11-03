# -*- coding: utf-8 -*-
"""
train_model.py (5 points) 
When this is called using python train_model.py in the command line, 
this will take in the training dataset csv, perform the necessary data cleaning 
and imputation, and fit a classification model to the dependent Y. 
There must be data check steps and clear commenting for each step inside the .py file.
The output for running this file is the random forest model saved as a .pkl file 
in the local directory. Remember that the thought process and decision for 
why you chose the final model must be clearly documented in this section. 
eda.ipynb (0 points)

Learn data analysis from 
https://github.com/minsuk-heo/kaggle-titanic/blob/master/titanic-solution.ipynb
http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
@author: Ada
"""

#import packages
import pandas as pd
import os


#data import
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()

#Sample size of the train data are 891 rows and 12 features
train.shape
train.info()
train.isnull().sum()

#Feature Engineering - data clearning, formatting and transforming
#Features: Transforming the object type values to numerical data

#Title: get from "Name" colunm and map title to numeriacal data
train_test_data = [train, test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

train['Title'].value_counts()

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    


#Sex: map to numeriacal data
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


#Age: 
#1.fill missing age with median age for each title (Mr, Mrs, Miss, Others)
#2. classify age by each 10-year for adualt  
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

for dataset in train_test_data:
    dataset['AgeBK'] = dataset['Age']

for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <=16, 'Age'] = 0
    dataset.loc[ (dataset['Age'] >16) & (dataset['Age'] <= 26), 'Age'] = 1
    dataset.loc[ (dataset['Age'] >26) & (dataset['Age'] <= 36), 'Age'] = 2
    dataset.loc[ (dataset['Age'] >36) & (dataset['Age'] <= 62), 'Age'] = 3
    dataset.loc[ (dataset['Age'] >62), 'Age'] = 4


#Pclass (Ticket class by a proxy for socio-economic status (SES)): 
#1 = 1st (Upper) , 2 = 2nd (Middle), 3 = 3rd (Lower)
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']


#Embarked(Port of Embarkation): C = Cherbourg, Q = Queenstown, S = Southampton    
#map to numerical data
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')   
    
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)   


#Fare: fill missing Fare with median fare for each Pclass
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
train.head(50)


#Cabin:
#1. take first letter of the data as the clasification
#2. fill missing Fare with median fare for each Pclass
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
    
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']

cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

#remove unuseful features 
train.drop('Name', axis = 1, inplace=True)
test.drop('Name', axis = 1, inplace = True)

features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)

train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape
train_data.info()


#modeling
# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np

#1. Cross Validation (K-fold)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

#kNN
clf = KNeighborsClassifier(n_neighbors = 13)
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring='accuracy')
round(np.mean(score)*100, 2) # kNN Score

#2.Decision Tree
clf = DecisionTreeClassifier()
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring='accuracy')
round(np.mean(score)*100, 2) # decision tree Score

#3.Ramdom Forest
clf = RandomForestClassifier(n_estimators=13)
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring='accuracy')
round(np.mean(score)*100, 2)  # Random Forest Score

#4.Naive Bayes
clf = GaussianNB()
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring='accuracy')
round(np.mean(score)*100, 2)  # Naive Bayes Score

#5.SVM
clf = SVC()
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring='accuracy')
round(np.mean(score)*100,2)



#Since Ramdom Forest has the best score, apply for this model to test data set.
#1.kNN: 72.39
#2.decision tree: 78.23
#3.Random Forest: 80.92
#4.Naive Bayes: 76.09
#5.SVC : 71.72

#Model input/output
import pickle

# Save the best model to file in the current working directory
clf = RandomForestClassifier(n_estimators=13) 
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring='accuracy')
clf.fit(train_data, target)

pkl_filename = "model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(clf, file)
    

