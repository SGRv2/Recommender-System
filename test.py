# import pandas as pd
# import numpy as np
import argparse
from preprocess import preprocess_UB_CF, preprocess_content_boosted
from test_run import UB_CF_test

from utils import *
from IO_utils import *


"""
1. preprocess data
2. split ratings
3. run target user predictions
"""

def main():
  
  # parse args for the file names
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', action='store', dest='input_file', help='Input file')
  parser.add_argument('--output', action='store', dest='output_file', help='Output file')
  parser.add_argument('--content_boost', action='store_true', dest='content_boosted', help='Boolean for running content boost')
  parser.set_defaults(content_boosted=False)
  results = parser.parse_args()

  # read data an test users
  ratings, movies = read_data("ratings.csv", "movies.csv")
  test_users = read_test_users(results.input_file)

  # split data according to test users
  ratings_train, ratings_test = split_ratings_data(ratings, test_users)

  print("Running preprocessing")
  preprocess = preprocess_content_boosted(ratings, ratings_train) if results.content_boosted else preprocess_UB_CF(ratings, ratings_train)
  topN = 10
  
  print("Running predictions on all unseen movies")
  test_run = UB_CF_test(ratings_test, preprocess.utility_matrix, 
                        preprocess.weighted_similarity_matrix if results.content_boosted else preprocess.similarity_matrix, 
                        preprocess.average_ratings, preprocess.movieId_map, topN)
  predictions = test_run.predict()
  
  # get top5 recommended for unseen movies and top 5 rated before 
  rec_top5 = get_rec_top5(predictions)
  rated_top5 = highest_rated_seen_movies(ratings_test)
  print ("Starting write")
  write_output(results.output_file, ratings_test, rec_top5, rated_top5, predictions, preprocess, movies)


if __name__ == "__main__":
  main()