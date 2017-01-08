# -*- coding: utf-8 -*-
"""
Created on Fri Dec 02 03:42:20 2016

@author: JUSTIN
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#load training and testing set
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Examine training set structure
print(train.head())
print(train.describe())
print(train.shape)

#Replace NaN values
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

train["Embarked"] =train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")

test.Fare[152] = test["Fare"].median()

#Recode Sex
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

#Recode Embarked
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

#Create Child column
train["Child"] = float('NaN')

train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0

#Create family size column
train["family_size"] = train["SibSp"] + train["Parch"] + 1
test["family_size"] = test["SibSp"] + test["Parch"] + 1

#Decision Tree

#Create numpy arrays for features and prediction
target = train["Survived"].values
features_forest = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked", "family_size"]].values

#Fit Decision Tree
forest = RandomForestClassifier(max_depth = 10, min_samples_split = 2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)


#Check Results of decision tree
print(my_forest.score(features_forest, target))


#Predict Testing Data
test_features = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked", "family_size"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest)

#Create new Data Frame for export
#PassengerId = np.array(test["PassengerId"]).astype(int)
#my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
#print(my_solution)

#Write CSV file