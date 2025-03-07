# !/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# (c) Copyright University of Southampton, 2023
#
# Copyright in this software belongs to University of Southampton,
# Highfield, University Road, Southampton SO17 1BJ
#
# Example created for COMP3208 Coursework
######################################################################

import numpy as np
import logging,sys,codecs,math

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')

#
# training function for cosine similarity recommender system algorithm
#
def train_model( train_data = None ) :

	#
	# calculate size of training set ignoring any zero rating entries
	# n_users = number of users
	# n_items = number of items
	#
	n_users = 100
	n_items = 10

	#
	# initialize vector and matrix ready for population
	#
	user_average_rating_vector = np.zeros( n_users, dtype=np.float16 )
	item_sim_matrix = np.full( (n_items, n_items), fill_value=-1.0, dtype=np.float16 )

	#
	# describe each chunk of code used per processing step clearly.
	# for example for cosine similarity explain the equation used to compute the item to item similarity weights,
	# adding in an explaination of the maths behind algorithms as its used (you need to provide evidence of a deep understanding)
	#
	# e.g.
	#   compute the target variable using a linear function.
	#   the linear function used is
	#     x = 2 * y
	#   where
	#     x = target variable
	#     y = input variable
	#

	# blah blah ... self-documented code ... blah blah

	#
	# return the populated (a) item to item simularity matrix and (b) user average rating vector
	#
	return user_average_rating_vector, item_sim_matrix


#
# infer function for cosine similarity recommender system algorithm
#
def predict_ratings( test_data = None, item_sim_matrix = None, user_average_rating_vector = None ) :

	#
	# calculate size of training set ignoring any zero rating entries
	# n_users = number of users
	# n_items = number of items
	#
	n_users = len( user_average_rating_vector )
	n_items = item_sim_matrix.shape[0]

	#
	# initialize vector and matrix ready for population
	#
	predictions = np.zeros((n_users, n_items), dtype=np.float16)

	#
	# describe each chunk of code used per processing step clearly.
	# for example for cosine similarity explain the equation used to compute the item to item similarity weights,
	# adding in an explaination of the maths behind algorithms as its used (you need to provide evidence of a deep understanding)
	#
	# e.g.
	#   compute the target variable using a linear function.
	#   the linear function used is
	#     x = 2 * y
	#   where
	#     x = target variable
	#     y = input variable
	#

	# blah blah ... self-documented code ... blah blah

	#
	# return the predicted ratings matrix
	#
	return predictions


#
# serialize a set of predictions to file
#
def serialize_predictions( output_file = None, prediction_matrix = None ) :
	#
	# describe each chunk of code used per processing step clearly.
	# for example for cosine similarity explain the equation used to compute the item to item similarity weights,
	# adding in an explaination of the maths behind algorithms as its used (you need to provide evidence of a deep understanding)
	#
	# e.g.
	#   compute the target variable using a linear function.
	#   the linear function used is
	#     x = 2 * y
	#   where
	#     x = target variable
	#     y = input variable
	#

	# blah blah ... self-documented code ... blah blah

	return

#
# load a set of ratings from file
#
def load_data( file = None ) :
	#
	# describe each chunk of code used per processing step clearly.
	# for example for cosine similarity explain the equation used to compute the item to item similarity weights,
	# adding in an explaination of the maths behind algorithms as its used (you need to provide evidence of a deep understanding)
	#
	# e.g.
	#   compute the target variable using a linear function.
	#   the linear function used is
	#     x = 2 * y
	#   where
	#     x = target variable
	#     y = input variable
	#

	# blah blah ... self-documented code ... blah blah

	n_users = 10
	n_items = 100
	return np.zeros((n_users, n_items), dtype=np.float16)


if __name__ == '__main__':

	logger.info( 'Example of self documented code for a COMP3208 submission' )

	#
	# load test and training data into memory
	# INPUT = test and training files
	# OUTPUT = (user,item) rating matrix for test and training data
	#
	train = load_data( file = 'comp3208_100k_train_withratings.csv' )
	test = load_data( file = 'comp3208_100k_test_withoutratings.csv' )

	#
	# call the train function to learn similarity weights for cosine similarity recommender system algorithm
	# INPUT = (user,item) rating matrix loaded from training dataset
	# OUTPUT = matrix (item to item simularity); vector (user average rating)
	#
	item_sim_matrix, user_average_rating_vector = train_model( train_data = train )

	#
	# call the infer function to execute the cosine similarity recommender system algorithm
	# INPUT = (user,item) rating matrix loaded from test dataset; matrix (item to item simularity); vector (user average rating)
	# OUTPUT = (user,item) rating prediction matrix
	#
	pred = predict_ratings( test_data = test, item_sim_matrix = item_sim_matrix, user_average_rating_vector = user_average_rating_vector )

	#
	# serialize the rating predictions to file ready for upload to ECS handin system
	# INPUT = output CSV filename; (user,item) rating prediction matrix
	# OUTPUT = None (predictions serialized to file)
	#
	serialize_predictions( output_file = 'submission.csv', prediction_matrix = pred )

	logger.info( 'Predictions saved to file submission.csv' )
