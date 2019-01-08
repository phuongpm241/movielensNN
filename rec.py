'''
Tested on: Python 3.5.2, numpy 1.14.1, pandas 0.20.3

Please read README for info about the dataset.

**Please note that all code here is optional -- feel free to
use a completely different implementation.**
'''
import numpy as np
import pandas as pd

np.random.seed(42)

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
    Y = np.zeros((num_users, num_movies)) 
    
    rows, _ = data.shape
    
    for i in range(rows):
        userid, movieid, rate = data[i,:]
        Y[map_users[userid]][map_movies[movieid]] = rate 
    
    return Y
    
    
    
def MSE_SV_threshold(A_hat, data_test, map_users, map_movies):
    error = 0
    
    for i in range(len(data_test)):
        u, m, r = data_test[i,:]
        error += (A_hat[map_users[u]][map_movies[m]] - r)**2
    
    return error




def accuracy_SV_threshold(A_hat, data_test, map_users, map_movies):
    """The error metrics to measure accuracy
       The prediction in A_hat is rounded to the nearest integer
    """
    
    correct = 0
    for i in range(len(data_test)):
        u, m, r = data_test[i,:]
        correct += int(round(A_hat[map_users[u]][map_movies[m]])) == r
        
    return correct/ len(data_test)


def SV_thresholding_hard(data_train, data_test, map_users, map_movies, tau_h=2):
    
    num_users = len(map_users)
    num_movies = len(map_movies)
    
    Y_prime = get_rating_matrix(data_train, map_users, map_movies) # TODO
    U, sigma, V = SVD(Y_prime)

    # TODO: truncate SVs using hard thresholding
    
    keep = (sigma >= tau_h).astype(int)
    new_sigma = sigma*keep
    new_S = np.diag(new_sigma)
    
    known = np.count_nonzero(Y_prime)
    print('known element', known)
    p_hat = known/(num_users*num_movies) #Proportion of known elements
    A_hat = 1/p_hat*np.dot(U, np.dot(new_S, V))
    
    #np.savetxt("SVhard.csv", A_hat, delimiter=",")

    # TODO: Evaluate using some error metric measured on test set
    train_acc = accuracy(A_hat, data_train, map_users, map_movies)
    test_acc = accuracy(A_hat, data_test, map_users, map_movies)
    return train_acc, test_acc

def SV_thresholding_soft(data_train, data_test, map_users, map_movies, tau_s=2):
    Y_prime = get_rating_matrix(data_train, map_users, map_movies) # TODO
    U, sigma, V = SVD(Y_prime)

    # TODO: truncate SVs using soft thresholding
    p_hat = None
    A_hat = None 

    # TODO: Evaluate using some error metric measured on test set
    error = None # TODO
    return error


def SVD(Y):
    # TODO: compute SV decomposition of a matrix Y
    
    u, s, vh = np.linalg.svd(Y, full_matrices = False)
    return u, s, vh

def cosine_distance(vs_from_u, user_a, user_b):
#    a_rated = vs_from_u[user_a]
#    b_rated = vs_from_u[user_b]
#    
#    M = []
#    
#    for i in a_rated.keys():
#        if i in b_rated:
#            M.append(i)
#    
#    # Use the common rating set M, build y_a and y_b
    pass
    
    
    
    
    
    
    

def similarity_matrix(data_train):
    """ Calculate the cosine similarity between two users (undirected graph)
    """
#    n = max(d[0] for d in data_train)+1 # users
#    m = max(d[1] for d in data_train)+1
#    
#    Y_hat = np.zeros((n, n))
#    
#    vs_from_u = [dict() for a in range(n)]  # AI (a-index set)
#    
#    for (a, i, r) in data_train:
#        vs_from_u[a][i] = r
#    
#    for a in range(n):
#        for b in range(i, n):
#            # Find all the movies which a and b have rated
    pass
        
            
            
            
            
            
    
    
    
    
    
    

def CF_user_user(data_train, data_test, map_users, map_movies, k=1):
    

    sim = None # TODO

    # TODO: choose an aggregation method to predict unknown ratings

    # TODO: Evaluate using some error metric measured on test set
    error = None # TODO
    return error


def neural_network(data_train, data_test):
    pass # TODO: use a neural network to predict ratings. Open-ended!


if __name__ == '__main__':
    users = get_user_data()[0].values # Array of users ID
    movies = get_movie_data()[0].values # Array of movies ID
    
    map_users = {u : i for i, u in enumerate(users)}
    map_movies = {m : i for i, m in enumerate(movies)}
    
    data_train, data_test = get_rating_data()
    
    
#    print(SV_thresholding_hard(data_train, data_test, map_users, map_movies))

    # Singular Value Thresholding
    # SV_thresholding(data_train, data_test)

    # Alternating Least Squares
#    print(ALS(data_train, data_test))

    # Collaborative Filtering
    # CF_user_user(data_train, data_test)

    # Neural Network
    # neural_network(data_train, data_test)

    # TODO: compare performance of different models
    # TODO determine effect of hyperparameters in each model