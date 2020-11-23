
import pandas as pd
import numpy as np


"""
Intialised by : ratings, ratings_train dataframes
Precomputes : utility matrix, similarity matrix, average ratings per user  
"""
class preprocess_UB_CF():
  def __init__(self, ratings, ratings_train):
    self.ratings = ratings
    self.ratings_train = ratings_train
    self.movieId_map = self.calc_movieId_map()
    self.inv_movieId_map = {v: k for k, v in self.movieId_map.items()}
    
    self.num_users = self.ratings.userId.unique().shape[0]
    self.num_movies = ratings.movieId.unique().shape[0]
    
    self.test_size = 10 # hard coded for now
    self.train_size = self.num_users - self.test_size 

    self.utility_matrix = self.get_utility_matrix(self.ratings)
    self.utility_matrix_train = self.get_utility_matrix(self.ratings_train)

    self.similarity_matrix = self.get_similarity_matrix(self.utility_matrix_train)
    self.average_ratings = self.get_average_ratings(self.utility_matrix_train) 
    
  def calc_movieId_map(self):
    index = 0
    movieId_map = {}
    for movieId in self.ratings.movieId:
      if movieId not in movieId_map.keys():
        movieId_map[movieId] = index
        index+=1
    return movieId_map

  def get_utility_matrix(self, ratings):
    utility_matrix = np.zeros((self.num_users, self.num_movies))
    # setting values in utility matrix
    for record in self.ratings.itertuples():
      _, user, movie, rating = record
      utility_matrix[(user - 1), self.movieId_map[movie]] = rating
    return utility_matrix

  def get_similarity_matrix(self, utility_matrix):
    utility_df = pd.DataFrame(data=self.utility_matrix.transpose())
    return utility_df.corr()
  
  def get_average_ratings(self, utility_matrix):
    average_ratings = {}
    for user in range(self.utility_matrix.shape[0]):
      util_matrix_user = self.utility_matrix[user]
      util_matrix_user[util_matrix_user == 0] = np.nan
      average_ratings[user] = np.nanmean(util_matrix_user)
    return average_ratings


"""
Inherits the preprocess class and adds content boosting 

Intialised by : ratings, ratings_train dataframes
Precomputes : utility matrix, weighted similarity matrix, average ratings per user  
"""
class preprocess_content_boosted(preprocess_UB_CF):
  def __init__(self, ratings, ratings_train):
    super().__init__(ratings, ratings_train)
    self.movies_rated_per_user = self.get_movies_rated_per_user() 
    self.harmonic_means = self.get_harmonic_means()
    self.f1 = lambda x: self.func1(x)
    self.f2 = lambda x: self.func2(x)
    self.sig_wt = self.get_sig_wt(self.utility_matrix)
    self.hybrid_corr_wt = self.get_hybrid_corr_wts(self.harmonic_means, self.sig_wt)
    self.weighted_similarity_matrix = self.get_weighted_similarity_matrix(self.hybrid_corr_wt)

  """
  calculate harmonic mean matrix of m values for each pair of users
  """
  def get_harmonic_means(self):
    num_users = self.movies_rated_per_user.shape[0]
    harmonic_means = np.zeros((num_users, num_users))
    for u1 in range(num_users):
      for u2 in range(num_users):
        if (u1 == u2): continue
        m1 = self.movies_rated_per_user['m'][u1]
        m2 = self.movies_rated_per_user['m'][u2]
        harmonic_means[u1][u2] =  (2*m1*m2) / (m1 + m2)
    
    return harmonic_means

  # computes m values and no. of movies rated per user 
  def get_movies_rated_per_user(self):
    data = []
    cols = ['userId', 'num_movies']
    u = self.ratings.userId.unique()
    for user in u:
      data.append([user,self.ratings.userId.value_counts()[user]])
    mu = pd.DataFrame(data, columns=cols)
    mu['m'] = [1 if mu['num_movies'][i] >= 50 else mu['num_movies'][i] / 50 for i in range(mu.shape[0])] 
    return mu

  def func1(self, val):
    if val>0:
      return 1
    else:
      return 0

  def func2(self, val):
    if val > 50:
      return 1.0
    else:
      return float(val/50)

  def get_sig_wt(self, utility_mat): # find the co-rated items
    x = np.nan_to_num(utility_mat)
    vfunc1 = np.vectorize(self.f1)
    x = vfunc1(x)  # convert utility mat to binary matrix 
    cooccurrence_matrix = np.dot(x, x.transpose())
    vfunc2 = np.vectorize(self.f2)
    sig_wt = vfunc2(cooccurrence_matrix) 
    return sig_wt

  def get_hybrid_corr_wts(self, harmonic_means, sig_wt):
    return harmonic_means + sig_wt

  def get_weighted_similarity_matrix(self, hybrid_corr_wt):
    return (self.similarity_matrix*hybrid_corr_wt)