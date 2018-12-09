import numpy as np
import pandas as pd

EPSILON = 0.000001

def single_iteration(ratings_graph, biases, true_ratings, alpha, beta):
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

    converged = True
    if ((true_ratings is not None) and np.any(np.abs(true_ratings - next_true_ratings) > EPSILON)) or \
        np.any(np.abs(biases - next_biases) > EPSILON):
        converged = False

    return converged, next_true_ratings, next_biases

def single_iteration_user_ent(ratings_graph, biases, true_ratings, alpha, beta):
    # Update Rating
    graph_shape = ratings_graph.get_graph_shape()
    indiv_true_ratings = np.maximum(np.zeros(graph_shape), np.minimum(np.ones(graph_shape),
                                        ratings_graph.get_user_entity_ratings() - alpha * biases))
    rating_denoms = ratings_graph.get_entity_rating_counts()
    next_true_ratings = np.sum(ratings_graph.get_user_entity_adj_matrix() * indiv_true_ratings, axis=0) / rating_denoms

    # Update Biases
    indiv_biases = (ratings_graph.get_user_entity_adj_matrix()*(ratings_graph.get_user_entity_ratings() - next_true_ratings)).dot(ratings_graph.get_entity_sim())
    bias_denoms = (ratings_graph.get_user_entity_adj_matrix()).dot(ratings_graph.get_entity_sim())
    next_biases = (1-beta)*biases + beta*(indiv_biases) / bias_denoms

    converged = True
    if ((true_ratings is not None) and np.any(np.abs(true_ratings - next_true_ratings) > EPSILON)) or \
        np.any(np.abs(biases - next_biases) > EPSILON):
        converged = False

    return converged, next_true_ratings, next_biases


def debias_ratings_baseline(ratings_graph, initial_alpha, decay_rate, max_iters, beta, user_entity_specific=False):
    np.random.seed(10)
    ground_truth_ratings = ratings_graph.get_ground_truth_ratings()
    true_ratings = [np.random.uniform((ratings_graph.num_entities,))]
    if not user_entity_specific:
        biases = [np.random.uniform(low = -1, high = 1, size = (ratings_graph.num_users,))]
    else:
        biases = [np.random.uniform(low = -1, high = 1, size = (ratings_graph.num_users, ratings_graph.num_entities))]
    errors = []

    converged = False
    num_iters = 0
    alpha = initial_alpha
    while not converged and num_iters < max_iters:
        true_rate_or_none = None if not true_ratings else true_ratings[-1]
        if not user_entity_specific:
            iter_out = single_iteration(ratings_graph, biases[-1], true_rate_or_none, alpha, beta)
        else:
            iter_out = single_iteration_user_ent(ratings_graph, biases[-1], true_rate_or_none, alpha, beta)

        converged, next_true_ratings, next_biases = iter_out
        true_ratings.append(next_true_ratings)
        biases.append(next_biases)
        if ground_truth_ratings is not None:
            errors.append(np.sqrt(np.mean((next_true_ratings - ground_truth_ratings)**2)))
        num_iters += 1
        alpha = alpha/decay_rate

    return biases, true_ratings, errors
