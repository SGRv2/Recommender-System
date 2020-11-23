import pandas as pd
import numpy as np
import matplotlib.pylab as plt

"""
Model to generate neighborhood and then calculate predicted ratings using the resnick prediction formula
Called for each fold of the k folds
Initialised by : ratings train dataframe, movieId map, utility_matrix and average ratings.
"""
class UB_CF:
  def __init__(self, ratings_train, movieId_map, utility_matrix, similarity_matrix, average_ratings):
    self.ratings_train = ratings_train
    self.movieId_map = movieId_map
    self.utility_matrix = utility_matrix
    self.similarity_matrix = similarity_matrix
    # average ratings over all seen movies for each user
    self.average_ratings = average_ratings

    self.MAE = []
    self.neighbors = []
    self.num_users = self.utility_matrix.shape[0]

  """
  Resnick prediction function
  divide_by_two : optional boolean to control for overflowing ratings above 5
  """
  def get_resnick(self, user, movie, divide_by_two = True):
    resnick = 0
    sum_sim = 0 
    for nbr in self.neighbors[user]:
      nbr_av = self.average_ratings[nbr]
      if ((self.utility_matrix[nbr][movie])!=0 and not np.isnan(self.utility_matrix[nbr][movie])):
        sum_sim += np.abs(self.similarity_matrix[user][nbr])  # sum of absolute values
        resnick += ((self.utility_matrix[nbr][movie]-nbr_av)*self.similarity_matrix[user][nbr])

    resnick = resnick/sum_sim if sum_sim!=0 else 0
    resnick = resnick/2 if divide_by_two else resnick
    user_av = self.average_ratings[user]
    resnick += user_av
    
    return resnick
  

  # generate neighborhood for all users
  def generate_neighborhood(self, topN):
    self.neighbors = [self.get_neighbors(topN, self.similarity_matrix[user]) for user in range(self.num_users)]   

  # generates neighbor list for a user given k and corresponding row of similarity matrix
  def get_neighbors(self, k, similarity_matrix_tuple):
    sim_tuple = np.array(similarity_matrix_tuple)
    neighbors = (-sim_tuple).argsort()[1:k+1]
    return neighbors


  """
  Given list of choices of topN, plots list of kfold Mean average errors 
  """  
  def plot_MAE(self, topN_choices):
    average_topN_MAE = {}
    k = 0
    for i in range(0, len(self.MAE), 5):
      av = 0
      for j in range(5):
        av += self.MAE[i+j]
      average_topN_MAE[topN_choices[k]] = av / 5
      k+=1

    lists = sorted(average_topN_MAE.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    plt.plot(x, y)
    plt.show()
