#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 20:14:45 2018

@author: phuongpham
"""
import numpy as np
import pandas as pd


def get_user_data():
    '''Returns user info as a Pandas DataFrame,
    where rows are users and columns are the features
    UserID, Gender, Age, Occupation, Zip Code'''
    users_filename = "users.dat"
    return pd.read_csv(users_filename, header=None, sep='::', engine='python')


def get_movie_data():
    '''Returns movie info as a Pandas DataFrame,
    where rows are movies and columns are the features
    MovieID, Title, Genres'''
    movies_filename = "movies.dat"
    return pd.read_csv(movies_filename, header=None, sep='::', engine='python')


def get_rating_data():
    '''Returns rating info as two 2D numpy arrays:
    one for training and one for testing.
    Rows are ratings and columns are
    UserID, MovieID, Rating, Timestamp'''
    ratings_filename = "ratings.dat"
    df = pd.read_csv(ratings_filename, header=None, sep='::', engine='python')
    data = df.ix[:,:2].values
    np.random.shuffle(data)

    train_length = int(df.shape[0] * .8)

    train = data[:train_length]
    test = data[train_length:]
    return train, test


def get_rating_matrix(data, map_users, map_movies):
    '''Given a 2D numpy arrays containing UserID, MovieID, and Rating,
    returns a 2D numpy array Y where Y[UserID][MovieID] = Rating for all entries
    in df, with all other elements equal to None.'''
    
    num_users = len(map_users)
    num_movies = len(map_movies)
    A = np.zeros((num_users, num_movies)) 
    mask = np.zeros((num_users, num_movies)) 
    
    rows, _ = data.shape
    
    for i in range(rows):
        userid, movieid, rate = data[i,:]
        A[map_users[userid]][map_movies[movieid]] = rate 
        mask[map_users[userid]][map_movies[movieid]] = 1
    
    return A, mask

def MSE_SV_threshold(A_hat, data_test, map_users, map_movies):
    error = 0
    
    for i in range(len(data_test)):
        u, m, r = data_test[i,:]
        error += (A_hat[map_users[u]][map_movies[m]] - r)**2
    
    return np.sqrt(error/len(data_test))


def svt_solve(A, mask, tau=None, delta=None, epsilon=1e-2, max_iterations=200):
  """
  Solve using iterative singular value thresholding.
  [ Cai, Candes, and Shen 2010 ]
  Parameters:
  -----------
  A : m x n array
    matrix to complete
  mask : m x n array
    matrix with entries zero (if missing) or one (if present)
  tau : float
    singular value thresholding amount;, default to 5 * (m + n) / 2
  delta : float
    step size per iteration; default to 1.2 times the undersampling ratio
  epsilon : float
    convergence condition on the relative reconstruction error
  max_iterations: int
    hard limit on maximum number of iterations
  Returns:
  --------
  X: m x n array
    completed matrix
  """
  Y = np.zeros_like(A)

  if not tau:
    tau = 5 * np.sum(A.shape) / 2
  if not delta:
    delta = 1.2 * np.prod(A.shape) / np.sum(mask)

  for _ in range(max_iterations):  
    try:
        U, S, V = np.linalg.svd(Y, full_matrices=False)
    
        S = np.maximum(S - tau, 0)
    
        X = np.linalg.multi_dot([U, np.diag(S), V])
        Y += delta * mask * (A - X)
    
        rel_recon_error = np.linalg.norm(mask * (X - A)) / np.linalg.norm(mask * A)
        if _ % 10 == 0:
          print("Iteration: %i; Rel error: %.4f" % (_ + 1, rel_recon_error))
        if rel_recon_error < epsilon:
          break
      except:
          print("SVD did not converge")
          return X

  return X

users = get_user_data()[0].values # Array of users ID
movies = get_movie_data()[0].values # Array of movies ID

map_users = {u : i for i, u in enumerate(users)}
map_movies = {m : i for i, m in enumerate(movies)}

data_train, data_test = get_rating_data()

A, mask = get_rating_matrix(data_train, map_users, map_movies)
A_hat = svt_solve(A, mask, delta = 1)
print(MSE_SV_threshold(A_hat, data_test, map_users, map_movies))