
####################################################################################################
# Supervised Random Walk
# Author: Wei Zhang
# 06/21/2017
#
# v0.2 added L-FGBS algorithm for the training 07/26/2017
#
# v0.3 added 3 options loss=('squared', 'absolute'), norm_type=('L2','L1'), and 
# maximize_diff=(False, True) 07/27/2017
#
# Reference: https://arxiv.org/pdf/1011.4071.pdf
# However, this package has a different objective function:
# J = lam*||w||**2 + sum_over_u_and_v(y*sum((pu-pv)**2))
# where pu and pv are the propagation scores of 2 samples, 
# and y (1, 0) or (1, -1) indicates whether the 2 samples belong to the same group.
# Thus the goal is to minimize the difference of samples within each group
# (and optionally maximize the difference between different groups)
####################################################################################################

import numpy as np
import os
from collections import Counter
import functools
from scipy.optimize import fmin_l_bfgs_b

# Read edges and edge features from a file
# Return edges (e by 2) and features (e by w), 
# and gene2index (optionally written into a file)
# Note: (i, j) and (j, i) should be 2 lines in the input file, 
# with different features
def load_network(file_name, output_dir=''):
    # Load a graph
    print "* Loading network..."
    gene_pairs = []
    features = []
    gene_set = set()
    with open(file_name) as file_handle:
        for line in file_handle.read().splitlines():
            row = line.split('\t')
            gene_pairs.append([row[0],row[1]])
            features.append(row[2:])
            gene_set.add(row[0])
            gene_set.add(row[1])
            
    index_map = ''
    gene2index = {}
    for i, gene in enumerate(sorted(list(gene_set))):
        gene2index[gene] = i
        index_map += '{}\t{}\n'.format(i, gene)
        
    if output_dir:
        # Set up an output directory
        print "* Saving node list to file..."
        os.system( 'mkdir -p ' + output_dir )
        with open("{}/index_genes".format(output_dir), 'w') as outfile:
            outfile.write( "\n".join(index_map) )
        
    edges = [[gene2index[i] for i in gene_pair] for gene_pair in gene_pairs]
    return edges, features, gene2index


# Return the edge strength (vector) calculated by a logistic function
# Inputs: edge features (e by w) and edge feature weights (vector w)
def logistic_edge_strength(features, w):
    return  1.0 / (1+np.exp(-np.dot(features, w)))


# Calculate the gradient of edge strength functioin with
# respect to edge feature weights, returns a matrix of gradients (e by w)
# Equation: dStrength/dw = features * edge_strength * (1-edge_strength)
def strength_gradient(features, edge_strength):
    features = np.matrix(features)
    logistic_slop = np.multiply(edge_strength, (1-edge_strength))[:,np.newaxis]
    nfeatures = features.shape[1]
    return np.multiply(features, np.repeat(logistic_slop, nfeatures, axis=1))


# Normalize a matrix by row sums,
# return a normalized matrix
def renorm(M):    
    rowSum = np.sum(M, axis=1)
    return np.array(M).astype(float)/rowSum[:,np.newaxis]


# This function takes edges (e by 2), edge features (e by w), 
# edge feature weights (vector w), and the number of nodes,
# then retruns a transition matrix Q (n by n), and its gradient Q_grad (n by n)
# with respect to edge feature weights
# Note: (i, j) and (j, i) should be 2 rows in edges, with different features
def generate_Q_and_gradient(edges, nnodes, features, w):
    # Calculate edge strength and the gradient of strength
    edge_strength = logistic_edge_strength(features, w)
    strength_grad = strength_gradient(features, edge_strength)
    # M_strength (n by n) is a matrix containing edge strength
    # where M[i,j] = Strength[i,j];
    # M_strength_grad (w by n by n) contains the gradient of strength
    # where M[l,i,j] = dStrength[i,j]/dw[l]
    M_strength = np.zeros((nnodes, nnodes))
    nfeatures = len(w)
    M_strength_grad = np.zeros((nfeatures, nnodes, nnodes))
    for i in range(len(edges)):
        M_strength[edges[i][0],edges[i][1]] = edge_strength[i]
        for l in range(nfeatures):
            M_strength_grad[l,edges[i][0],edges[i][1]] = strength_grad[i,l]

    # Normalize the transition matrix
    Q = renorm(M_strength)
    
    # Calculate the gradient of Q: Q_grad (w by n by n)
    # Equation: Q[i,j] = strength[i,j] / sum_over_k(strength[i,k]), thus
    # dQ[i,j]/dw[l] = (dStrength[i,j]/dw[l]*(sum_over_k(strength[i,k])) 
    #                   -Strength[i,j]*(sum_over_k(dStrength[i,k]/dw[l])))
    #                  / (sum_over_k(strength[i,k]))**2
    M_strength_rowSum = np.sum(M_strength, axis=1)
    M_strength_grad_rowSum = np.sum(M_strength_grad, axis=2)
    # Inflate M_strength_rowSum so that it has the dimention (w by n by n)
    M_strength_rowSum = M_strength_rowSum[np.newaxis,:,np.newaxis]
    M_strength_rowSum = np.repeat(M_strength_rowSum, nnodes, axis=2)
    M_strength_rowSum = np.repeat(M_strength_rowSum, nfeatures, axis=0)
    # Inflate M_strength_grad_rowSum so that it has the dimention (w by n by n)
    M_strength_grad_rowSum = M_strength_grad_rowSum[:,:,np.newaxis]
    M_strength_grad_rowSum = np.repeat(M_strength_grad_rowSum, nnodes, axis=2)
    # Inflate M_strength so that it has the dimention (w by n by n)
    M_strength = M_strength[np.newaxis,:,:]
    M_strength = np.repeat(M_strength, nfeatures, axis=0)
    # Calculate Q_grad
    Q_grad = (np.multiply(M_strength_grad, M_strength_rowSum) 
              -np.multiply(M_strength, M_strength_grad_rowSum)) / np.square(M_strength_rowSum)
    
    return Q, Q_grad


####################################################################################################
# *** The fllowing 2 functions calculate Q and unweighted Q, 
# without calculating the gradient
def generate_Q(edges, nnodes, features, w):
    M_strength = np.zeros((nnodes, nnodes))
    # Calculate edge strength and the gradient of strength
    edge_strength = logistic_edge_strength(features, w)
    # M_strength (n by n) is a matrix containing edge strength
    # where M[i,j] = Strength[i,j];
    for i in range(len(edges)):
        M_strength[edges[i][0],edges[i][1]] = edge_strength[i]
    # Normalize the transition matrix
    Q = renorm(M_strength)
    return Q
def generate_Q_unweighted(edges, nnodes):
    M_strength = np.zeros((nnodes, nnodes))
    for i in range(len(edges)):
        M_strength[edges[i][0],edges[i][1]] = 1
    # Normalize the transition matrix
    Q = renorm(M_strength)
    return Q
####################################################################################################


# This function takes a normalized transition matrix Q (n by n), 
# initial state P_init (m by n), and reset probability,
# then use iteration to find the personalized PageRank at convergence.
def iterative_PPR(Q, P_init, rst_prob):
    # Normalize P_init by row sums
    P_init = renorm(P_init)
    
    # Takes P_init and a transition matrix to find the PageRank of nodes
    P = P_init.copy()
    P_new =  (1-rst_prob)*np.dot(P, Q) + rst_prob*P_init
    while not(np.allclose(P, P_new)):
        P = P_new
        P_new =  (1-rst_prob)*np.dot(P, Q) + rst_prob*P_init
    return P_new


# This function takes PPR scores P (m by n), transition matrix Q (n by n), 
# and partial derivatives of Q with respect to edge feature weights Q_grad (w by n by n), 
# and iteratively calculate the gradient of PPR scores P_grad (w by m by n)
# with respect to edge feature weights until converged
# Equation: P = (1-rst_prob)*P.*Q + rst_prob*P_init
# therefore dP/dw = (1-rst_prob) * (dP/dw.*Q+P.*dQ/dw)
def iterative_P_gradient(P, Q, Q_grad, rst_prob):
    # Initlalize P_grad to be all zeros. See below for P_grad_1_iter
    P_grad = np.zeros(tuple([Q_grad.shape[0]])+P.shape)
    P_grad_new = P_grad_1_iter(rst_prob, P_grad, Q, P, Q_grad) 
    # Iteratively calculate P_grad until converged
    while not(np.allclose(P_grad, P_grad_new)):
        P_grad = P_grad_new
        P_grad_new = P_grad_1_iter(rst_prob, P_grad, Q, P, Q_grad)
    return P_grad_new
# *This function calculate P_grad for one iteration, which is called by iterative_P_gradient
def P_grad_1_iter(rst_prob, P_grad, Q, P, Q_grad):
    P_grad_new = P_grad.copy()
    for i in range(P_grad.shape[0]):
        P_grad_new[i,:,:] = (1-rst_prob) * (np.dot(P_grad[i,:,:], Q)+np.dot(P, Q_grad[i,:,:]))
    return P_grad_new


# This is the cost function
# Input: 2 lines of propagated scores pu and pv
# (return a scalar)
def cost_func(pu, pv, loss='squared'):
    if loss == 'squared':
        # return sum((pu-pv)**2)
        return np.sum(np.square(pu-pv))
    elif loss == 'absolute':
        # return sum(|pu-pv|)
        return np.sum(np.abs(pu-pv))


# This is the gradient of the cost function
# with respect to (pu-pv)
# Input: 2 lines of propagated scores pu and pv
# (return a vector (length = n))
def cost_func_gradient(pu, pv, loss='squared'):
    if loss == 'squared':
        # return 2 * (pu-pv)
        return 2 * (pu-pv)
    elif loss == 'absolute':
        # return sign(pu-pv)
        return np.sign(pu-pv)


# This is the objective function to be minimized
# Supervised Random Walk PageRank scores are calculated based on 
# input network and features, then cost function value is calculated 
# according to the derived pagerank scores. 
# (returns a scalar)
# Equation: J = lam*||w||**2 + sum_over_u_and_v(y*sum((pu-pv)**2)/z_uv)
# where y = 1 if u and v belong to the same group, otherwise y = 0 (or -1);
# z_uv is the total number of comparisons in the group where u and v belongs to
# Inputs: P (m by n), group labels (length m vector), 
# regularization parameter labmda, and edge feature weights (vector w)
def obj_func(P, group_lables, group2npairs, lam, w, loss='squared', 
             norm_type='L2', maximize_diff=False):
    # Compute cost from PPR scores
    cost = 0
    npats = len(group_lables)
    for u in range(npats):
        for v in range(u+1, npats):
            if group_lables[u] == group_lables[v]:
                group = group_lables[u]
                y = 1
            else:
                if not maximize_diff:
                    continue
                group = 'diff'
                y = -1
            cost += cost_func(P[u,:], P[v,:], loss) / group2npairs[group] * y
    
    
    if norm_type == 'L2':
        # Calculate the L2 norm squared of edge feature weights
        norm = np.dot(w, w)
    elif norm_type == 'L1':
        # Calculate the L1 norm of edge feature weights
        norm = np.sum(np.abs(w))
        
    return lam*norm + cost


# This is the gradient of the objective function
# Equation: J_grad = lam*2*w 
#                    + sum_over_u_and_v(y*(pu_grad-pv_grad).*(2*(pu-pv))/z_uv)
# where y = 1 if u and v belong to the same group, otherwise y = 0 (or -1);
# z_uv is the total number of comparisons in the group where u and v belongs to
# Q, Q_grad, P, P_grad are provided to derive J_grad
# (J_grad is a vector with the length of w)
def obj_func_gradient(Q, Q_grad, P, P_grad, group_lables, group2npairs, lam, w, 
                      loss='squared', norm_type='L2', maximize_diff=False):
    # Compute the gradient of cost function with respect to edge feature weights w
    cost_grad = np.zeros(len(w))
    npats = len(group_lables)
    for u in range(npats):
        for v in range(u+1, npats):
            if group_lables[u] == group_lables[v]:
                group = group_lables[u]
                y = 1
            else:
                if not maximize_diff:
                    continue
                group = 'diff'
                y = -1
            # Calculate cost_func_grad (length n vector) 
            # which is the gradient of the cost function
            # with respect to (pu - pv)
            cost_func_grad = cost_func_gradient(P[u,:], P[v,:], loss)
            # Calculate cost_grad
            # Equation: cost_grad = (pu_grad-pv_grad) .* cost_func_grad
            # Dimension: (w by n) .* (n by 1) = (w by 1)
            cost_grad += np.dot((P_grad[:,u,:]-P_grad[:,v,:]),
                                cost_func_grad) / group2npairs[group] * y
    
    if norm_type == 'L2':
        # Calculate the gradient of the L2 norm squared term
        norm_grad = 2 * w
    elif norm_type == 'L1':
        # Calculate the gradient of the L1 term
        norm_grad = np.sign(w)
    
    return lam*norm_grad + cost_grad


# For each group, calculate the number of sample-pairs 
# for normalization in the objective function
def count_numbers_per_group(group_lables):
    group2nsamples = Counter(group_lables)
    group2npairs = {}
    sum_npairs_withingroup = 0
    for group in group2nsamples:
        group2npairs[group] = 0.5 * group2nsamples[group] * (group2nsamples[group]-1)
        sum_npairs_withingroup += group2npairs[group]
    nsamples = len(group_lables)
    group2npairs['diff'] = 0.5*nsamples*(nsamples-1) - sum_npairs_withingroup
    return group2npairs
    
####################################################################################################
# This part of code uses gradient descent to minimize the objective function
####################################################################################################

# This wrapper function sequentially calculate Q, Q_grad, P, P_grad to derive J and J_grad
def calculate_J_and_gradient(edges, features, nnodes, P_init, rst_prob, group_lables, lam, w, 
                             loss='squared', norm_type='L2', maximize_diff=False):
    # Generate transition matrix Q (n by n), and its gradient Q_grad (w by n by n)
    # according to edge features and weights
    Q, Q_grad = generate_Q_and_gradient(edges, nnodes, features, w)
    # Calculate Personalized PageRank (PPR)
    P = iterative_PPR(Q, P_init, rst_prob)
    # Calculate the gradient of PPR (w by m by n)
    P_grad = iterative_P_gradient(P, Q, Q_grad, rst_prob)
    # For each group, calculate the number of sample-pairs 
    # to normalize in the objective function
    group2npairs = count_numbers_per_group(group_lables)
    # Calculate objective function J (scalar), 
    # and its gradient J_grad (length w vector)
    J = obj_func(P, group_lables, group2npairs, lam, w, loss, norm_type, maximize_diff)
    J_grad = obj_func_gradient(Q, Q_grad, P, P_grad, group_lables, group2npairs, lam, w, 
                               loss, norm_type, maximize_diff)
    return J, J_grad


# This function trains the edge feature weights of Supervised Random Walk
# using gradient descent functions
# Inputs: edges (e by 2), edge features (e by w), number of nodes,
# initial state P_init (m by n), reset probability, group labels (length m vector), 
# regularization parameter labmda, 
# standard deviation of initial edge feature weights, learning rate, 
# and the function for updating parameters
def train_SRW_GD(edges, features, nnodes, P_init, rst_prob, group_lables, lam,
                 w_init_sd=0.01, loss='squared', norm_type='L2', maximize_diff=False, 
                 learning_rate=0.1, update_w_func='Adam', **kwargs):
    # t is the iteration counter
    t = 0
    # Initialize edge feature weights from a Gaussian distribution, 
    # with standard deviation w_init_sd
    w = np.random.normal(scale = w_init_sd, size = features.shape[1])
    # Calculate the initlal J, and J_grad
    J, J_grad = calculate_J_and_gradient(edges, features, nnodes, P_init, 
                                         rst_prob, group_lables, lam, w, 
                                         loss, norm_type, maximize_diff)
    print t, 'iteration: J is ', J
    # Initialize the velocity of w, used for momentum methods
    v = np.zeros(w.shape)
    # Initialize the first and second moment estimate of J_grad, used for Adam
    m = np.zeros(w.shape)
    n = np.zeros(w.shape)
    # Initialize the momentum schecule PI_mu_t, used for Nadam
    PI_mu_t = np.array([1.])
    
    t += 1
    # Update w for the first time
    update_w(update_w_func, w, J_grad, learning_rate, m, n, t, PI_mu_t, v, **kwargs)
    # Update J and J_grad for the first time
    J_new, J_grad = calculate_J_and_gradient(edges, features, nnodes, P_init, 
                                             rst_prob, group_lables, lam, w, 
                                             loss, norm_type, maximize_diff)
    print t, 'iteration: J is ', J_new
    
    while not np.allclose(J, J_new):
        t += 1
        J = J_new
        # Update w
        update_w(update_w_func, w, J_grad, learning_rate, m, n, t, PI_mu_t, v, **kwargs)
        # Update J and J_grad
        J_new, J_grad = calculate_J_and_gradient(edges, features, nnodes, P_init, 
                                                 rst_prob, group_lables, lam, w, 
                                                 loss, norm_type, maximize_diff)
        print t, 'iteration: J is ', J_new
    
    return w


# This is a wrapper function for updating parameters, called by train_SRW_GD
# It changes w, m, n, PI_mu_t, v in place
def update_w(func, w, J_grad, learning_rate, m, n, t, PI_mu_t, v, **kwargs):
    if func == 'Nadam':
        w[:], m[:], n[:], PI_mu_t[:] = update_w_Nadam(w, J_grad, learning_rate, m, n, 
                                                      t, PI_mu_t, **kwargs)
    elif func == 'Adam':
        w[:], m[:], n[:] = update_w_Adam(w, J_grad, learning_rate, m, n, t, **kwargs)
    elif func == 'Nesterov':
        w[:], v[:] = update_w_Nesterov_momentum(w, J_grad, learning_rate, v, **kwargs)
    elif func == 'momentum':
        w[:], v[:] = update_w_momentum(w, J_grad, learning_rate, v, **kwargs)


# *** Below are four functions for parameter updates:

# This function uses momentum to accelerate gradient descent
def update_w_momentum(w, J_grad, learning_rate, v, momentum=0.9):
    v = momentum*v - learning_rate*J_grad
    w += v
    return w, v

# This is a better version of momentum (Nesterov 1983). 
# See Geoffrey Hinton's lecture notes for details:
# http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
def update_w_Nesterov_momentum(w, J_grad, learning_rate, v, momentum = 0.9):
    v_prev = v.copy()
    v = momentum*v - learning_rate*J_grad
    # w is the position where we made a big (gamble) jump last time, then we
    # 1. move w back to where it was: -momentum*v_prev
    # 2. follow the new v vector to where it should be: +v
    # 3. make another gamble: +momentum*v
    # so the returned value w is actually w_ahead, 
    # which should be close to w when learning converges
    w += -momentum*v_prev +v +momentum*v
    return w, v

# Adam(Adaptive Moment Estimation)
# See here for details: 
# http://cs231n.github.io/neural-networks-3/#ada
def update_w_Adam(w, J_grad, learning_rate, m, n, t, eps=1e-8, beta1=0.9, beta2=0.999):
    m = beta1*m + (1-beta1)*J_grad #  first moment estimate
    m_hat = m / (1-beta1**t) # bias-corrected first moment estimate
    n = beta2*n + (1-beta2)*np.square(J_grad) # second moment estimate
    n_hat = n / (1-beta2**t) # bias-corrected second moment estimate
    w += -learning_rate*m_hat / (np.sqrt(n_hat)+eps)
    return w, m, n

# Nadam (Nesterov-accelerated adaptive moment estimation)
# http://cs229.stanford.edu/proj2015/054_report.pdf
def update_w_Nadam(w, J_grad, learning_rate, m, n, t, PI_mu_t, eps=1e-8, beta1=0.99, beta2=0.999):
    # Calculate the momentum schecule
    mu_t = beta1 * (1-0.5*0.96**(t/250))
    mu_next = beta1 * (1-0.5*0.96**((t+1)/250))
    PI_mu_t = PI_mu_t * mu_t
    PI_mu_next = PI_mu_t * mu_next
    
    J_grad_hat = J_grad / (1-PI_mu_t)
    m = beta1*m + (1-beta1)*J_grad # same as Adam
    m_hat = m / (1-PI_mu_next)
    n = beta2*n + (1-beta2)*np.square(J_grad) # same as Adam
    n_hat = n / (1-beta2**t) # same as Adam
    m_bar = mu_next*m_hat + (1-mu_t)*J_grad_hat # Nesterov
    w += -learning_rate*m_bar / (np.sqrt(n_hat)+eps)
    return w, m, n, PI_mu_t


####################################################################################################
# This part of code uses L-FGBS to minimize the objective function
####################################################################################################

# This wrapper function sequentially calculate Q and P to derive J
def calculate_J(edges, features, nnodes, P_init, rst_prob, group_lables, group2npairs, lam, w, 
                loss='squared', norm_type='L2', maximize_diff=False):
    # Generate transition matrix Q (n by n) according to edge features and weights
    Q = generate_Q(edges, nnodes, features, w)
    # Calculate Personalized PageRank (PPR)
    P = iterative_PPR(Q, P_init, rst_prob)
    # Calculate objective function J (scalar)
    J = obj_func(P, group_lables, group2npairs, lam, w, loss, norm_type, maximize_diff)
    return J


# This wrapper function sequentially calculate Q, Q_grad, P, P_grad to derive J and J_grad
def calculate_J_gradient(edges, features, nnodes, P_init, rst_prob, group_lables, group2npairs, 
                         lam, w, loss='squared', norm_type='L2', maximize_diff=False):
    # Generate transition matrix Q (n by n), and its gradient Q_grad (w by n by n)
    # according to edge features and weights
    Q, Q_grad = generate_Q_and_gradient(edges, nnodes, features, w)
    # Calculate Personalized PageRank (PPR)
    P = iterative_PPR(Q, P_init, rst_prob)
    # Calculate the gradient of PPR (w by m by n)
    P_grad = iterative_P_gradient(P, Q, Q_grad, rst_prob)
    # Calculate the gradient of the objective function J_grad (length w vector)
    J_grad = obj_func_gradient(Q, Q_grad, P, P_grad, group_lables, group2npairs, lam, w, loss, 
                               norm_type, maximize_diff)
    return J_grad


# This function trains the edge feature weights of Supervised Random Walk using L-BFGS
# Inputs: edges (e by 2), edge features (e by w), number of nodes,
# initial state P_init (m by n), reset probability, group labels (length m vector), 
# regularization parameter labmda, and standard deviation of initial edge feature weights
def train_SRW_BFGS(edges, features, nnodes, P_init, rst_prob, group_lables, lam, w_init_sd=0.01, 
                   loss='squared', norm_type='L2', maximize_diff=False):
    # Initialize edge feature weights from a Gaussian distribution, 
    # with standard deviation w_init_sd
    w_init = np.random.normal(scale = w_init_sd, size = features.shape[1])
    # For each group, calculate the number of sample-pairs 
    # to normalize in the objective function
    group2npairs = count_numbers_per_group(group_lables)
    
    # scipy's L-BFGS-B optimizer is called to iteratively optimize J
    # J and J_grad are the main input to the BFGS optimizer
    # functools.partial is used to generate partial functions of J and J_grad 
    # with only one free argument w
    w, J, d = fmin_l_bfgs_b(functools.partial(calculate_J, edges, features, nnodes, P_init, 
                                              rst_prob, group_lables, group2npairs, lam, 
                                              loss=loss, norm_type=norm_type, 
                                              maximize_diff=maximize_diff), 
                            w_init,
                            fprime = functools.partial(calculate_J_gradient, edges, features, 
                                                       nnodes, P_init, rst_prob, group_lables, 
                                                       group2npairs, lam, loss=loss, 
                                                       norm_type=norm_type, 
                                                       maximize_diff=maximize_diff))
    
    return w, J, d


