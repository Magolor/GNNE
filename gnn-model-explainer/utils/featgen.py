""" featgen.py

Node feature generators.

"""
import networkx as nx
import numpy as np
import random

import abc


class FeatureGen(metaclass=abc.ABCMeta):
    """Feature Generator base class."""
    @abc.abstractmethod
    def gen_node_features(self, G):
        pass

class BinomialFeatureGen(FeatureGen):
    """Binomial Feature class."""
    def __init__(self, num, p, seed=998244353):
        np.random.seed(seed); self.mask = [True] * int(p*num) + [False] * (num-int(p*num)); np.random.shuffle(self.mask)

    def gen_node_features(self, G):
        feat_dict = {i:{'feat': np.array([1,1,1,1,0,0,0,0,1,1] if self.mask[i] else [0,0,0,0,1,1,1,1,1,1], dtype=np.float32)} for i in G.nodes()}
        nx.set_node_attributes(G, feat_dict)

class CorrelatedFeatureGen(FeatureGen):
    """Correlated Feature class."""
    def __init__(self, struct, normal, p, seed=998244353):
        np.random.seed(seed); assert(sorted(struct + normal) == [v for v in range(len(struct)+len(normal))])
        struct_feat = struct; np.random.shuffle(struct_feat); struct_feat = struct_feat[:int(p*len(struct))]
        normal_feat = normal; np.random.shuffle(normal_feat); normal_feat = normal_feat[:len(struct)-int(p*len(struct))]
        self.mask = [(v in struct_feat or v in normal_feat) for v in range(len(struct)+len(normal))]

    def gen_node_features(self, G):
        feat_dict = {i:{'feat': np.array([1,1,1,1,0,0,0,0,1,1] if self.mask[i] else [0,0,0,0,1,1,1,1,1,1], dtype=np.float32)} for i in G.nodes()}
        nx.set_node_attributes(G, feat_dict)

class ConstFeatureGen(FeatureGen):
    """Constant Feature class."""
    def __init__(self, val):
        self.val = val

    def gen_node_features(self, G):
        feat_dict = {i:{'feat': np.array(self.val, dtype=np.float32)} for i in G.nodes()}
        nx.set_node_attributes(G, feat_dict)

class GaussianFeatureGen(FeatureGen):
    """Gaussian Feature class."""
    def __init__(self, mu, sigma):
        self.mu = mu
        if sigma.ndim < 2:
            self.sigma = np.diag(sigma)
        else:
            self.sigma = sigma

    def gen_node_features(self, G):
        feat = np.random.multivariate_normal(self.mu, self.sigma, G.number_of_nodes())
        feat_dict = {
                i: {"feat": feat[i]} for i in range(feat.shape[0])
            }
        nx.set_node_attributes(G, feat_dict)


class GridFeatureGen(FeatureGen):
    """Grid Feature class."""
    def __init__(self, mu, sigma, com_choices):
        self.mu = mu                    # Mean
        self.sigma = sigma              # Variance
        self.com_choices = com_choices  # List of possible community labels

    def gen_node_features(self, G):
        # Generate community assignment
        community_dict = {
            n: self.com_choices[0] if G.degree(n) < 4 else self.com_choices[1]
            for n in G.nodes()
        }

        # Generate random variable
        s = np.random.normal(self.mu, self.sigma, G.number_of_nodes())

        # Generate features
        feat_dict = {
            n: {"feat": np.asarray([community_dict[n], s[i]])}
            for i, n in enumerate(G.nodes())
        }

        nx.set_node_attributes(G, feat_dict)
        return community_dict

