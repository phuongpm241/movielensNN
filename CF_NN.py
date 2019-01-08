#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 19:28:22 2018

@author: phuongpham
"""

# CFModel.py
#
# A simple implementation of matrix factorization for collaborative filtering
# expressed as a Keras Sequential model. This code is based on the approach
# outlined in [Alkahest](http://www.fenris.org/)'s blog post
# [Collaborative Filtering in Keras](http://www.fenris.org/2016/03/07/collaborative-filtering-in-keras).
#
# License: MIT. See the LICENSE file for the copyright notice.
#

import numpy as np
import pandas as pd
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense, Concatenate
from keras.models import Sequential

def get_rating_data():
    '''Returns rating info as two 2D numpy arrays:
    one for training and one for testing.
    Rows are ratings and columns are
    UserID, MovieID, Rating, Timestamp'''
    ratings_filename = "ratings.dat"
    df = pd.read_csv(ratings_filename, header=None, sep='::', engine='python')
    data = df.ix[:,:2].values
#    np.random.shuffle(data)
#    train_length = int(df.shape[0] * .8)
#    train = data[:train_length]
#    test = data[train_length:]
    return data


class CFModel(Sequential):

    def __init__(self, n_users, m_items, k_factors, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, k_factors, input_length=1))
        P.add(Reshape((k_factors,)))
        Q = Sequential()
        Q.add(Embedding(m_items, k_factors, input_length=1))
        Q.add(Reshape((k_factors,)))
        super(CFModel, self).__init__(**kwargs)
        self.add(Merge([P, Q], mode='dot', dot_axes=1))

    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]

class DeepModel(Sequential):

    def __init__(self, n_users, m_items, k_factors, p_dropout=0.1, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, k_factors, input_length=1))
        P.add(Reshape((k_factors,)))
        Q = Sequential()
        Q.add(Embedding(m_items, k_factors, input_length=1))
        Q.add(Reshape((k_factors,)))
        super(DeepModel, self).__init__(**kwargs)
        self.add(Merge([P, Q], mode='concat'))
#        self.add(Concatenate([P, Q]))
        self.add(Dropout(p_dropout))
        self.add(Dense(100, activation='relu'))
        self.add(Dropout(p_dropout))
        self.add(Dense(10, activation='relu'))
        self.add(Dropout(p_dropout))
        self.add(Dense(5, activation='softmax'))

    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]
    
if __name__ == '__main__':
    data = get_rating_data()
    users = data[:,0]
    movies = data[:,1]
    ratings = data[:,2]
    one_hot_ratings = np.eye(5)[ratings - 1]
    
    baseline_model = DeepModel(max(users)+1, max(movies)+1, 10)
    baseline_model.compile(loss='mse', optimizer='adamax')
    baseline_model.fit(x = [users, movies], y = one_hot_ratings, batch_size = 64,\
                       epochs = 10, validation_split=0.2)
