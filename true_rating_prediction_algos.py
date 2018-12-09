import numpy as np
import pandas as pd

EPSILON = 0.000001

def single_iteration(ratings_graph, biases, true_ratings, alpha, beta):
    # Remove nans
    biases = np.nan_to_num(biases)
    true_ratings = np.nan_to_num(true_ratings)
    
    # Update Ratings
    graph_shape = ratings_graph.get_graph_shape()
    indiv_true_ratings = np.maximum(np.zeros(graph_shape), np.minimum(np.ones(graph_shape),
                                        ratings_graph.get_user_entity_ratings() - alpha * biases[:, None]))
    rating_denoms = ratings_graph.get_entity_rating_counts()
    next_true_ratings = np.sum(ratings_graph.get_user_entity_adj_matrix() * indiv_true_ratings, axis=0) / rating_denoms

    # Update Biases
    indiv_biases = ratings_graph.get_user_entity_ratings() - next_true_ratings
    bias_denoms = ratings_graph.get_user_rating_counts()
    next_biases = (1-beta)*biases + beta*(np.sum(ratings_graph.get_user_entity_adj_matrix() * indiv_biases, axis=1) / bias_denoms)
    
    if (np.isnan(biases).any() or np.isnan(true_ratings).any()):
        import pdb; pdb.set_trace()
    
    converged = True
    if ((true_ratings is not None) and np.any(np.abs(true_ratings - next_true_ratings) > EPSILON)) or \
        np.any(np.abs(biases - next_biases) > EPSILON):
        converged = False

    return converged, next_true_ratings, next_biases

def mishra_prediction(ratings_graph, initial_alpha=0.99, \
                                     decay_rate=1.00, \
                                     max_iters = 200000, \
                                     beta = 0.1, \
                                     multi_dim_bias=False, \
                                     seed=10):
    np.random.seed(seed)
    ground_truth_ratings = ratings_graph.get_ground_truth_ratings()
    true_ratings = [np.random.uniform((ratings_graph.num_entities,))]
    if not multi_dim_bias:
        biases = [np.random.uniform(low = -1, high = 1, size = (ratings_graph.num_users,))]
    else:
        biases = [np.random.uniform(low = -1, high = 1, size = (ratings_graph.num_users, ratings_graph.num_entities))]
    errors = []

    converged = False
    num_iters = 0
    alpha = initial_alpha
    while not converged and num_iters < max_iters:
        true_rate_or_none = None if not true_ratings else true_ratings[-1]
        if not multi_dim_bias:
            iter_out = single_iteration(ratings_graph, biases[-1], true_rate_or_none, alpha, beta)
        else:
            iter_out = single_iteration_multi_dim_bias(ratings_graph, biases[-1], true_rate_or_none, alpha, beta)

        converged, next_true_ratings, next_biases = iter_out
        true_ratings.append(next_true_ratings)
        biases.append(next_biases)
        if ground_truth_ratings is not None:
            errors.append(np.sqrt(np.mean((next_true_ratings - ground_truth_ratings)**2)))
        num_iters += 1
        alpha = alpha/decay_rate

    return biases, true_ratings, errors

def mean_prediction(ratings_graph):
    user_entity_ratings = ratings_graph.get_user_entity_ratings()
    user_entity_adj_matrix = ratings_graph.get_user_entity_adj_matrix()
    user_entity_adj_matrix_na = user_entity_adj_matrix.copy()
    user_entity_adj_matrix_na[user_entity_adj_matrix==0]=np.nan

    mean_pred = np.nanmean(user_entity_ratings*(user_entity_adj_matrix_na), axis=0)
    return mean_pred
                  
def median_prediction(ratings_graph):
    user_entity_ratings = ratings_graph.get_user_entity_ratings()
    user_entity_adj_matrix = ratings_graph.get_user_entity_adj_matrix()
    user_entity_adj_matrix_na = user_entity_adj_matrix.copy()
    user_entity_adj_matrix_na[user_entity_adj_matrix==0]=np.nan
    
    median_pred = np.nanmedian(user_entity_ratings*(user_entity_adj_matrix_na), axis=0)
    return median_pred

def get_pred_error(pred, ratings_graph):
    error = np.sqrt(np.mean((pred - ratings_graph.get_ground_truth_ratings())**2))
    return error