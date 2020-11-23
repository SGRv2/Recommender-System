import argparse
import pandas as pd
import numpy as np

from utils import *
from IO_utils import *


def main():
  
  # parse args for the file names
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', action='store', dest='input_file', default='ratings.csv', help='Input file')
  parser.add_argument('--output', action='store', dest='eval_file', default='eval.csv', help='Output file')
  parser.add_argument('--content_boost', action='store_true', dest='content_boosted', help='Boolean for running content boost')
  parser.set_defaults(content_boosted=False)
  results = parser.parse_args()
  ratings, movies = read_data(results.input_file, "movies.csv")

  # read test users
  test_users = read_test_users("test_user.txt")
  # split data into train, test, leave test untouched till test_run
  ratings_train, ratings_test = split_ratings_data(ratings, test_users)

  # run kfold cross-validation
  MAE = KFold_evaluate(ratings, ratings_train, results.content_boosted)

  # output MAE to file
  print ("\nDone with Evaluation\nWriting to eval file\n")
  write_to_file(MAE, results.eval_file)




if __name__ == "__main__":
  main()