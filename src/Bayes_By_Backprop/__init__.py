"""Bayesian NN with Bayes By Backprop
"""
import torch.nn.functional as F

EPS = 1e-6

def w_to_std(w, beta=1, threshold=20):
    std_w = EPS + F.softplus(w, beta=beta, threshold=threshold)
    return std_w

def sample_weights(W_mu, b_mu, W_p, b_p):
    """Quick method for sampling weights and exporting weights
    
    Sampling W from N(W_mu, std_w^2) as follows:
        eps_W ~ N(0, 1^2)
        std_w = 1e-6 + log(1+exp(W_p)) (if W_p > 20, std_w = 1e-6 + W_p)
        W = W_mu + 1 * std_w * eps_W

    Sampling b from N(b_mu, std_b^2) as follows:
        eps_b ~ N(0, 1^2)
        std_b = 1e-6 + log(1+exp(b_p)) (if b_p > 20, std_w = 1e-6 + b_p)
        b = b_mu + 1 * std_b * eps_b

    This function samples b only if b_mu is not `None`
    """
    eps_W = W_mu.data.new(W_mu.size()).normal_()
    # sample parameters
    std_w = w_to_std(W_p)
    W = W_mu + 1 * std_w * eps_W

    if b_mu is not None:
        std_b = w_to_std(b_p)
        eps_b = b_mu.data.new(b_mu.size()).normal_()
        b = b_mu + 1 * std_b * eps_b
    else:
        b = None

    return W, b
