import os
import csv
import pandas as pd
import numpy as np
from utils import get_movie_title_from_id, get_movie_title

"""
Input : Mean Average Error across 5 folds, file to write eval data to 
Writes data using csv writer 
"""
def write_to_file(MAE, eval_filename):
  try:
    os.remove(eval_filename)
  except OSError:
    pass
  with open(eval_filename, mode='w') as eval_file:
      eval_writer = csv.writer(eval_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

      eval_writer.writerow(['Cross Val Step', 'Mean Absolute Error'])
      for i,mae in enumerate(MAE):
        eval_writer.writerow([i+1, mae])

"""
Reads test line separated user ids from the filename
""" 
def read_test_users(filename):
  test_file = pd.read_csv(filename, sep="\n")
  test_file = test_file.values.tolist()
  test_users = []
  for user in test_file:
    test_users.append(user[0])
  return test_users

"""
Reads ratings and movies into dataframes with the filenames as input
"""
def read_data(ratings_filepath, movies_filepath):
  ratings = pd.read_csv(ratings_filepath)
  ratings = ratings.drop(labels=['timestamp'], axis=1)
  movies = pd.read_csv(movies_filepath)
  return ratings, movies

"""
Writes the output of test user prediction in a tabular form into a csv file 
"""
def write_output(output_file, ratings_test, rec_top5, rated_top5, predictions, preprocess, movies):
  try:
    os.remove(output_file)
  except OSError:
    pass
  with open(output_file, mode='w') as output_file:
      output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

      output_writer.writerow(['Test User', 'Predicted Movie', 'Movies Seen in the Past'])
      output_writer.writerow(['', 'Movies, Rating', 'Movies, Rating >3'])
      for user in ratings_test.userId.unique():
        for i in range(5):
          pred_movie =  get_movie_title_from_id(movies, rec_top5[user-1][i], preprocess.inv_movieId_map)  # use top 5 recommended dict to get movie name
          pred_rating = round(predictions[user-1][rec_top5[user-1][i]], 2) # round off to 2 decimal places 
          seen_movie = get_movie_title(movies, rated_top5[user][i]) # get title
          movie_mapId = preprocess.movieId_map[rated_top5[user][i]] # get 0 indexed movieId
          seen_rating = round(preprocess.utility_matrix[user-1][movie_mapId], 2) # round off to 2 decimal places
          output_writer.writerow([user, (pred_movie, pred_rating), (seen_movie, seen_rating)]) # write row to file