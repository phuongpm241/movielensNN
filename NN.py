#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:39:14 2018

@author: phuongpham
"""
import numpy as np
import pandas as pd
from keras.layers.merge import concatenate
from keras.layers import Embedding, Input, Reshape, Dropout, Dense
from keras.models import Model

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

def get_baseline_model(n_users, n_items, emb_dim=10, p_dropout = 0.1):
    #input layers
    user_input = Input(shape=[1], name='user_inp')
    item_input = Input(shape=[1], name='item_inp')
    
    #embedding layers
    user_embedding = Embedding(output_dim=emb_dim, input_dim=n_users + 1,
                               input_length=1, name='user_emb')(user_input)

    item_embedding = Embedding(output_dim=emb_dim, input_dim=n_items + 1,
                               input_length=1, name='item_emb')(item_input)
    
    user_vecs = Reshape((emb_dim,))(user_embedding)
    item_vecs = Reshape((emb_dim,))(item_embedding)
    
    merge = concatenate([user_vecs, item_vecs])
    dense1 = Dense(100, activation='relu')(merge)
    dropout1 = Dropout(p_dropout)(dense1)
    dense2 = Dense(10, activation='relu')(dropout1)
    dropout2 = Dropout(p_dropout)(dense2)
    output = Dense(5, activation='softmax')(dropout2)

    #define model for training and inference
    model = Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer='adamax', loss='mse')
    
    return model


if __name__ == '__main__':
    data = get_rating_data()
    users = data[:,0]
    movies = data[:,1]
    ratings = data[:,2]
    one_hot_ratings = np.eye(5)[ratings - 1]
    
    baseline_model = get_baseline_model(max(users), max(movies))
    baseline_model.fit(x = [users, movies], y = one_hot_ratings, batch_size = 32,\
                       epochs = 10, validation_split=0.1)