
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

"""
Intialised by : ratings, ratings_train dataframes
Precomputes : utility matrix, similarity matrix, average ratings per user  
"""
class preprocess_UB_CF():
  def __init__(self, ratings, ratings_train):
    self.ratings = ratings
    self.ratings_train = ratings_train
    
    # generate mapping between dataset's ids to a 0 indexed movieId 
    self.movieId_map = self.calc_movieId_map()
    self.inv_movieId_map = {v: k for k, v in self.movieId_map.items()}
    
    self.num_users = self.ratings.userId.unique().shape[0]
    self.num_movies = ratings.movieId.unique().shape[0]
    
    self.test_size = 10 # hard coded for now
    self.train_size = self.num_users - self.test_size 

    # calculate utility_matix 
    self.utility_matrix = self.get_utility_matrix(self.ratings)
    # calulate only for train users for getting similarity matrix matrix 
    self.utility_matrix_train = self.get_utility_matrix(self.ratings_train)

    # calculate similarity matrix only for train users
    self.similarity_matrix = self.get_similarity_matrix(self.utility_matrix_train)
    self.average_ratings = self.get_average_ratings() 
    
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

  # df.corr() calculates the pearson similarities
  def get_similarity_matrix(self, utility_matrix):
    utility_df = pd.DataFrame(data=self.utility_matrix.transpose())
    return utility_df.corr()
  
  # returns average ratings on all rated movies of users
  def get_average_ratings(self):
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
  def __init__(self, ratings, ratings_train, movies):
    super().__init__(ratings, ratings_train)
    self.movies = movies
    self.movies_rated_per_user = self.get_movies_rated_per_user() 
    self.movie_genre, self.genre_map = self.get_movie_genres(self.movies)

    self.user_genre_counts, self.user_genre_vector = self.get_user_genres(self.movie_genre, self.genre_map, 
                                                                          self.utility_matrix, self.inv_movieId_map)

    
    self.user_genre_cosine_similarities = self.get_user_genre_cosine_similarities(self.utility_matrix, self.user_genre_vector)

    self.harmonic_means = self.get_harmonic_means()
    self.f1 = lambda x: self.func1(x)
    self.f2 = lambda x: self.func2(x)
    self.sig_wt = self.get_sig_wt(self.utility_matrix)
    
    self.hybrid_corr_wt = self.get_hybrid_corr_wts(self.harmonic_means, self.sig_wt)
    self.genre_boost = True
    if (self.genre_boost): self.hybrid_corr_wt += self.user_genre_cosine_similarities

    self.weighted_similarity_matrix = self.get_weighted_similarity_matrix(self.hybrid_corr_wt) 


  def get_user_genre_cosine_similarities(self, utm, user_genre_vector):
    user_genre_cs = np.zeros((utm.shape[0], utm.shape[0]))
    for u1 in range(utm.shape[0]):
      for u2 in range(utm.shape[0]):
        if (u1 == u2): user_genre_cs[u1][u2] = 1
        elif (u1 > u2): user_genre_cs[u1][u2] = user_genre_cs[u2][u1]
        else:
          user_genre_cs[u1][u2] = cosine_similarity(user_genre_vector[u1], user_genre_vector[u2])
    return user_genre_cs

  def get_movie_genres(self, movies):
    movie_genre = {}
    genres = {}
    j = 0
    for i in range(len(movies)):
      movie_genre[movies.iloc[i].movieId] = movies.iloc[i].genres.split('|')
      for g in movie_genre[movies.iloc[i].movieId]:
        if g not in genres:
          genres[g] = j
          j+=1
    return movie_genre, genres
  
  def get_user_genres(self, movie_genre, genre_map, utm, inv_movieId_map):
    user_genre_counts = {}
    num_genres = len(genre_map)

    for user in range(utm.shape[0]):
      user_genre_counts[user] = {}
      for movie in range(utm.shape[1]):
        if np.isnan(utm[user][movie]): continue
        movieId = inv_movieId_map[movie]
        for genre in movie_genre[movieId]:
          if genre not in user_genre_counts[user]:
            user_genre_counts[user][genre] = 0
          user_genre_counts[user][genre] += 1 

    user_genre_vector = {}
    for user in user_genre_counts.keys():
      user_genre_vector[user] = np.zeros((1,num_genres))
      for genre, count in user_genre_counts[user].items():
        a = 1
        user_genre_vector[user][0][genre_map[genre]] = count

    return user_genre_counts, user_genre_vector
  
  
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