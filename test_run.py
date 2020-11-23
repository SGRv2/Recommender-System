"""
1. Calculate similarities : Done
2. Get Neighbourhoods for test set only : Done
3. Make predictions on all unseen data : Done
"""
import pandas as pd
import numpy as np

"""
Generated neighborhoods and then predictions for all unseen movies for the test users


"""
class UB_CF_test():
  def __init__(self, ratings_test, utility_matrix, similarity_matrix, average_ratings, movieId_map, topN):
    self.ratings_test = ratings_test
    self.utility_matrix = utility_matrix
    self.average_ratings = average_ratings
    self.topN = topN
    self.movieId_map = movieId_map

    # self.similarity_matrix = self.get_similarity_matrix(utility_matrix)
    self.similarity_matrix = similarity_matrix 

    self.test_userIds = self.ratings_test.userId.unique()
    self.test_userIds -= 1
    self.num_test_users = len(self.test_userIds)
    
    self.neighbors = {}
    self.generate_neighborhood(self.topN, self.test_userIds)

  # # calculates pearson correlations for 
  # def get_similarity_matrix(self, utility_matrix):
  #   utility_df = pd.DataFrame(data=self.utility_matrix.transpose())
  #   return utility_df.corr()

  """
  For all unseen movies of a test user : predict the rating
  """
  def predict(self, ignore_unrated_movies = True):

    predictions = {}

    for user in self.test_userIds:
      print ("User {} \n".format(user))
      for mv, r in enumerate(self.utility_matrix[user]):
        if np.isnan(r):
          rating = self.get_resnick(user, mv)
          # skip if neighbors haven't rated the movie
          if (rating == -1 and ignore_unrated_movies): continue
          if user not in predictions:
            predictions[user] = {}
          predictions[user][mv] = rating
    
    return predictions

  # generate neighborhood for all users
  def generate_neighborhood(self, topN, test_userIds):
    for user in test_userIds:
      self.neighbors[user] = self.get_neighbors(topN, self.similarity_matrix[user])   

  # generates neighbor list for a user given k and corresponding row of similarity matrix
  def get_neighbors(self, k, similarity_matrix_tuple):
    sim_tuple = np.array(similarity_matrix_tuple)
    neighbors = (-sim_tuple).argsort()[1:k+1]
    return neighbors
 
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

    resnick = resnick/sum_sim if sum_sim!=0 else resnick
    resnick = resnick / 2 if divide_by_two else resnick
    user_av = self.average_ratings[user]
    if (resnick == 0): return -1
    resnick += user_av
    resnick = 5 if resnick > 5 else resnick
    return resnick