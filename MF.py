# Import libraries
import numpy as np
import pandas as pd
import argparse
import os

# Getting the required packages from the Surprise library
# Surprise is a Python scikit for building and analyzing recommender systems that deal with explicit rating data

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--ratings", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--movies", type=str,
	help="path to movies dataset")
args = vars(ap.parse_args())

# Reading ratings file
ratings = pd.read_csv(args["ratings"])

# Calculating the sparsity of the data
sparsity = round(1.0 - len(ratings) / float(n_users * n_movies), 3)
print('Sparsity Level of MovieLens dataset= ' +  str(sparsity * 100) + '%')

# Load Reader library
reader = Reader()

# Load ratings dataset with the Dataset library
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split the dataset for 5-fold evaluation
trainset, testset = train_test_split(data, test_size=.25)

algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)

accuracy.mae(predictions)
algo.predict(340, 544)
