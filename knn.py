#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 12:49:20 2018

@author: phuongpham
"""

import pandas as pd


from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline

#Load data
from surprise import Reader
from surprise import Dataset


#Validation
from surprise.model_selection import train_test_split
from surprise import accuracy


#Visualization
import matplotlib.pyplot as plt


#Load the dataframe dataset
ratings_filename = "ratings.dat"
df = pd.read_csv(ratings_filename, header=None, sep='::', engine='python')

#Convert into Surprise data structure
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df.ix[:,:2], reader)
trainset, testset = train_test_split(data, test_size=.2, random_state = 42)

# The first model to use is KNNBasic
vary_k = list(range(1, 41, 5))
test_acc = []
train_acc = []

for k in vary_k:
    model = KNNWithMeans(k = k, sim_options = {'name':'cosine', 'user_based': True})
    model.fit(trainset)
    test_predictions = model.test(testset)
    train_predictions = model.test(trainset.build_testset())
    
    test_acc.append(accuracy.rmse(test_predictions))
    train_acc.append(accuracy.rmse(train_predictions))
    print(">>> Finish k =", k)
    

plt.plot(vary_k, train_acc, label = 'Train RMSE')
plt.plot(vary_k, test_acc, label = 'Test RMSE')
plt.xlabel('k Nearest Neighbors')
plt.ylabel('RMSE')
plt.legend()
plt.show()







