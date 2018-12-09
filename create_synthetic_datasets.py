import numpy as np
from ratings_graph import RatingsGraph
import random
import matplotlib.pyplot as plt

def generate_biased_dataset(num_users, num_entities, gen_user_biases, gen_entity_features, num_features=1, seed=10, p_rate=0.2): 
    # The true ratings of the entities
    np.random.seed(seed)
    ground_truth_ratings = np.random.random(num_entities)
    
    # The number of features that define a movie
    # num_features = 1
    
    # Each user has a bias value along a certain feature dimension
    user_biases = np.zeros((num_users, num_features))
    for user_idx in range(num_users):
        for feature_idx in range(num_features):
            user_biases[user_idx][feature_idx] = gen_user_biases()
    
    # Setting the features for each entity along each feature dimension
    entity_features = np.zeros((num_entities, num_features))
    for entity_idx in range(num_entities):
        # Currently saying the maximum magnitude of the entity features is 1.0 / num_features
        #entity_features[entity_idx, :] = np.random.uniform(0, 1.0/num_features, num_features)
        entity_features[entity_idx, :] = gen_entity_features(num_features)
    
    # TODO: Think about how to intelligently normalize these features
    '''
    if num_features > 1:
        linfnorm = np.linalg.norm(entity_features, axis=1, ord=2)
        entity_features = entity_features.astype(np.float) / linfnorm[:,None]
    '''
    # Setting the user_item ratings and the user_item adjacency matrix
    user_entity_ratings = np.zeros((num_users, num_entities))
    user_entity_adj_matrix = np.zeros((num_users, num_entities))
    for user_idx in range(num_users):
        for entity_idx in range(num_entities):
            if np.random.random() < p_rate:
                user_entity_ratings[user_idx][entity_idx] = ground_truth_ratings[entity_idx] + \
                                                    (np.dot(entity_features[entity_idx, :], user_biases[user_idx,:]))
                user_entity_ratings[user_idx][entity_idx] = max(min(user_entity_ratings[user_idx][entity_idx], 1), 0)
                user_entity_adj_matrix[user_idx][entity_idx] = 1

    return user_entity_ratings, ground_truth_ratings, user_entity_adj_matrix, entity_features, user_biases