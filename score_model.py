# -*- coding: utf-8 -*-
"""
Predition using test set

@author: Ada
"""

#import packages
# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np
import pandas as pd

#Model input/output
import pickle

#data import
#train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test.head()

#Feature Engineering - data clearning, formatting and transforming
#Features: Transforming the object type values to numerical data

#Title: get from "Name" colunm and map title to numeriacal data
train_test_data = [test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

test['Title'].value_counts()

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
Pclass1 = test[test['Pclass']==1]['Embarked'].value_counts()
Pclass2 = test[test['Pclass']==2]['Embarked'].value_counts()
Pclass3 = test[test['Pclass']==3]['Embarked'].value_counts()
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
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test.head(50)


#Cabin:
#1. take first letter of the data as the clasification
#2. fill missing Fare with median fare for each Pclass
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
    
Pclass1 = test[test['Pclass']==1]['Cabin'].value_counts()
Pclass2 = test[test['Pclass']==2]['Cabin'].value_counts()
Pclass3 = test[test['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']

cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

#remove unuseful features 
test.drop('Name', axis = 1, inplace = True)

features_drop = ['Ticket', 'SibSp', 'Parch']
#train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
test_modify = test.drop(['PassengerId'], axis=1)
test_modify.head()


#prediction
# Load modle from file
pkl_filename = "model.pkl"
with open(pkl_filename, 'rb') as file:  
    pickle_model = pickle.load(file)

test_data = test.drop("PassengerId", axis=1).copy()

prediction = pickle_model.predict(test_data)  

submission = pd.DataFrame ( {
 "PassengerId": test["PassengerId"],
 "Survived": prediction
 } )
   
submission.head()

#Output Submission
submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')




