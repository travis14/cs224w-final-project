import numpy as np
import pandas as pd
from scipy.spatial import distance

class RatingsGraph:
    def __init__(self, user_entity_ratings, user_entity_adj_matrix, \
                ground_truth_ratings=None, entity_features=None):
        self.user_entity_ratings = user_entity_ratings # rows user, cols entities
        self.user_entity_adj_matrix = user_entity_adj_matrix
        if entity_features is not None:
            self.entity_features = entity_features
        if ground_truth_ratings is not None:
            self.ground_truth_ratings = ground_truth_ratings

        self.num_users = user_entity_ratings.shape[0]
        self.num_entities = user_entity_ratings.shape[1]

        self.entity_features = entity_features
        self.num_entity_features = entity_features.shape[1]

    def get_entity_features(self):
        return self.entity_features

    def get_graph_shape(self):
        return self.user_entity_ratings.shape

    def get_user_entity_ratings(self):
        return self.user_entity_ratings

    def get_user_entity_adj_matrix(self):
        return self.user_entity_adj_matrix

    def get_ground_truth_ratings(self):
        return self.ground_truth_ratings

    def get_entity_rating_counts(self):
        return np.sum(self.user_entity_adj_matrix, axis=0)

    def get_user_rating_counts(self):
        return np.sum(self.user_entity_adj_matrix, axis=1)
