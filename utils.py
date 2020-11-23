from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from heapq import nlargest

from preprocess import preprocess_UB_CF, preprocess_content_boosted
from models import UB_CF

import pandas as pd
import numpy as np

"""
Function responsile for running the kFold cross validation for 5 folds
Inputs : ratings dataframe, 
         ratings_train dataframe (after removing test users from ratings), 
         int topN - No. of neighbors to generate neighborhood for
         bool content_boosted to use weighted similarity if true or else pearson correlation

Outputs : list() Mean Average Error for the 5 folds   
"""
def KFold_evaluate(ratings, ratings_train, content_boosted, movies):
  MAE = [] # store mean absolute error for each run 
  if content_boosted: print ("Starting evaluation with content boosting")
  topN = 10 if content_boosted else 10
  kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
  j = 0 # track split number
  for training, testing in kf.split(ratings_train):
    j += 1
    print ("Split # {}".format(j))
    train = ratings_train.iloc[training]
    test = ratings_train.iloc[testing]
    # initialise preprocess class
    cv_preprocess = preprocess_content_boosted(ratings, train, movies) if content_boosted else preprocess_UB_CF(ratings, train) 
    # initialise model and generate neighborhood
    model = UB_CF(train, cv_preprocess.movieId_map, cv_preprocess.utility_matrix, 
                  cv_preprocess.weighted_similarity_matrix if content_boosted else cv_preprocess.similarity_matrix, 
                  cv_preprocess.average_ratings)
    model.generate_neighborhood(topN)


    predicted_ratings = np.zeros(test.rating.shape)
    count = 0
    for i,record in enumerate(test.itertuples()):
          _, user, movie, rating = record
          mv = cv_preprocess.movieId_map[movie] 

          try:
            pred = model.get_resnick(user-1, mv) # get prediction using resnick, 'user-1' here because users otherwise users indexed to 1 in ratings.csv
            predicted_ratings[i] = pred
          except Exception as e:
            # print(e)
            predicted_ratings[i] = 0
            count += 1  # number of exceptions 

    # print("Exception Count is : ", count)
    print("Mean Absolute Error is : ", mean_absolute_error(test.rating, predicted_ratings))
    MAE.append(mean_absolute_error(test.rating, predicted_ratings))

    print(MAE)

  return MAE

"""
Input : dict() of predictions per unseen movie for each of the test users 
Output : dict() of top 5 recommeded movies for each of the test users 
Uses a heap to get nlargest ratings from the dict 
"""
def get_rec_top5(pred):
  rec_top5 = {}
  for user in pred.keys():
    rec_top5[user] = nlargest(5, pred[user], key = pred[user].get)
  return rec_top5

"""
Splits the ratings dataframe according to list of test users
Output : train and test ratings dataframes
"""
def split_ratings_data(ratings, test_users):
  ratings_train = ratings[~ratings['userId'].isin(test_users)]
  ratings_test = ratings[ratings['userId'].isin(test_users)]
  return ratings_train, ratings_test

"""
Given 0 indexed movieId return original movieId found dataset
"""
def get_movie_title_from_id(movies, id, inv_movieId_map):
  id = inv_movieId_map[id]
  return get_movie_title(movies, id)

"""
Input : movies dataframe, original movieId
Output : title of the movie
"""
def get_movie_title(movies, id):
  ind = movies.index[movies['movieId'] == id].tolist()
  return movies.title[ind[0]]


"""
Input : test users' ratings dataframe
Output : top 5 rated movies seen before  
Sorts ratings and slices out top 5
"""
def highest_rated_seen_movies(ratings_te):
  sorted_by_ratings = ratings_te.groupby(["userId"]).apply(lambda x: x.sort_values(["rating"], ascending = False)).reset_index(drop=True)
  top5_seen = {}
  for user in ratings_te.userId.unique():
    user_start = sorted_by_ratings[sorted_by_ratings.userId == user].first_valid_index()  # res.movieId[user_start:user_start+5]
    top5_seen[user] = [mv for mv in sorted_by_ratings.movieId[user_start:user_start+5].to_numpy()]

  return top5_seen