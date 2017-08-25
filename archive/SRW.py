
####################################################################################################
# Supervised Random Walk
# Author: Wei Zhang
# 06/21/2017
####################################################################################################
#
# Reference: https://arxiv.org/pdf/1011.4071.pdf
# However, this package has a different objective function:
# J = lam*||w||**2 + sum_over_u_and_v(y*sum((pu-pv)**2))
# where pu and pv are the propagation scores of 2 samples, 
# and y (1, 0) or (1, -1) indicates whether the 2 samples belong to the same group.
# Thus the goal is to minimize the difference of samples within each group
# (and optionally maximize the difference between groups)
#
####################################################################################################
#
# v0.2.0 added L-FGBS algorithm for the training 07/26/2017
# v0.2.1 added 3 options loss=('squared', 'absolute'), norm_type=('L2', 'L1'), and 
# maximize_diff=(False, True) 07/27/2017
#
# v0.3.0 (current) use sparse matrix,  multiprocess.Pool and numba.jit whenever appropriate 
# Added an analytical solution in calculating the gradient of 'squared' loss
# Note: now 'squared' loss is much faster than the 'absolute' loss 08/02/2017
#
####################################################################################################

import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, issparse
import functools
from multiprocessing import Pool, cpu_count
from collections import Counter
from numba import jit
from scipy.optimize import fmin_l_bfgs_b

import time

####################################################################################################
# 1. The following three functions load a network, samples, and group labels from files
####################################################################################################

# Read edges and edge features from a file
# Return edges (e by 2) and features (e by w), 
# and node2index (optionally written into a file)
# (can also optionally add self-loops)
# Note: (i, j) and (j, i) should be 2 lines in the input file, 
# with different features
def load_network(file_name, output_dir='', add_selfloop=True):
    # Load a graph
    print "* Loading network..."
    df = pd.read_table(file_name)
    nfeatures = len(df.columns) - 2
    if add_selfloop:
        df['self_loop'] = 0.
    df['intercept'] = 1.
    node_set = set(df.iloc[:,0]) | set(df.iloc[:,1])
    
    node2index = {}
    index_map = ''
    selfloop_list = []
    for i, node in enumerate(sorted(list(node_set))):
        node2index[node] = i
        index_map += '{}\t{}\n'.format(i, node)
        # Add self-loops
        if add_selfloop:
            selfloop_list.append([node, node] + [0.]*nfeatures + [1., 1.])
    if add_selfloop:
        selfloop_df = pd.DataFrame(selfloop_list, columns=df.columns)
        df = pd.concat([df, selfloop_df])

    if output_dir:
        # Set up an output directory
        print "* Saving node list to file..."
        os.system('mkdir -p ' + output_dir)
        with open("{}/index_nodes".format(output_dir), 'w') as outfile:
            outfile.write( "\n".join(index_map) )
            
    edges = df.iloc[:,:2].applymap(lambda x: node2index[x]).as_matrix()
    features = csc_matrix(df.iloc[:,2:].as_matrix())
    return edges, features, node2index


# Read a sample by node matrix from a file
# Return the initial state of samples (m by n)
# and sample2index (optionally written into a file)
def load_samples(file_name, node2index, output_dir=''):
    df = pd.read_table(file_name, index_col=0)
    # df should be a sample by node matrix
    samples = df.index
    node_set = set(df.columns)&set(node2index.keys())
    print "\t- Nodes in adjacency matrix:", len(node_set)
    
    # Index mapping for samples
    sample2index = { samples[i]: i for i in range(len(samples)) }
    if output_dir:
        # Set up output directory
        print "* Saving sample list to file..."
        os.system( 'mkdir -p ' + output_dir)
        index_map = ["{}\t{}".format(i, samples[i]) for i in range(len(samples))]
        with open("{}/index_samples".format(output_dir), 'w') as outfile:
            outfile.write( "\n".join(index_map) )
    
    P_init = pd.DataFrame(index=df.index,columns=sorted(node2index.keys()))
    P_init.update(df)
    P_init = csr_matrix(P_init.fillna(0).as_matrix())
    return P_init, sample2index


# Read group labels from a file, return a list
def load_grouplabels(file_name):
    group_labels = []
    with open(file_name) as f:
        for line in f.read().rstrip().splitlines():
            row = line.split('\t')
            group_labels.append(row[1])
    return group_labels


####################################################################################################
# 2. The following functions calculate the objective function and a chain of gradients 
####################################################################################################

# Return the edge strength (vector) calculated by a logistic function
# Inputs: edge features (e by w) and edge feature weights (vector w)
def logistic_edge_strength(features, w):
    return  1.0 / (1+np.exp(-features.dot(w)))


# Calculate the gradient of edge strength functioin with
# respect to edge feature weights, returns a matrix of gradients (e by w)
# Equation: dStrength/dw = features * edge_strength * (1-edge_strength)
def strength_gradient(features, edge_strength):
    logistic_slop = np.multiply(edge_strength, (1-edge_strength))[:,np.newaxis]
    return features.multiply(logistic_slop)


# Normalize a matrix by row sums,
# return a normalized matrix
def renorm(M):
    return csr_matrix(M / M.sum(axis=1))


# This function takes edges (e by 2), edge features (e by w), 
# edge feature weights (vector w), and the number of nodes,
# then retruns a transition matrix Q (n by n), un-normalized
# transition matrix M_strength (n by n), row sums of M_strength (n by 1), 
# and the gradient of edge strength (e by w)
# Note: (i, j) and (j, i) should be 2 rows in edges, with different features
def strength_Q_and_gradient(edges, nnodes, features, w):
    # Calculate edge strength and the gradient of strength
    edge_strength = logistic_edge_strength(features, w)
    strength_grad = strength_gradient(features, edge_strength)
    # M_strength (n by n) is a matrix containing edge strength
    # where M[i,j] = strength[i,j];
    M_strength = csr_matrix((edge_strength, (edges[:,0], edges[:,1])), 
                            shape=(nnodes, nnodes))
    M_strength_rowSum = M_strength.sum(1)
    # Normalize the transition matrix
    Q = renorm(M_strength)
    return Q, M_strength, M_strength_rowSum, strength_grad


# This function generates a transition matrix Q (n by n)
# without calculating gradients
def generate_Q(edges, nnodes, features, w):
    # Calculate edge strength
    edge_strength = logistic_edge_strength(features, w)
    # M_strength (n by n) is a matrix containing edge strength
    # where M[i,j] = Strength[i,j];
    M_strength = csr_matrix((edge_strength, (edges[:,0], edges[:,1])), 
                            shape=(nnodes, nnodes))
    Q = renorm(M_strength)
    return Q

# *** This function calculates unweighted Q
def generate_Q_unweighted(edges, nnodes):
    M_strength = csr_matrix(([1.]*len(edges), (edges[:,0], edges[:,1])), 
                            shape=(nnodes, nnodes))
    # Normalize the transition matrix
    Q = renorm(M_strength)
    return Q


# Calculate the gradient of Q: Q_grad (n by n)
# with respect to one edge feature weight w[l]
# Equation: Q[i,j] = strength[i,j] / sum_over_k(strength[i,k]), thus
# dQ[i,j]/dw[l] = (dStrength[i,j]/dw[l]*(sum_over_k(strength[i,k])) 
#                  -Strength[i,j]*(sum_over_k(dStrength[i,k]/dw[l])))
#                 / (sum_over_k(strength[i,k]))**2
# Here strength_grad (e by 1) is with respect to one edge feature weight w[l]
def Q_gradient_1feature(edges, nnodes, M_strength, M_strength_rowSum, strength_grad):
    # M_strength_grad (n by n) is a matrix containing the gradient of edge strength
    # where M_strength_grad[i,j] = strength_grad[i,j];
    M_strength_grad = csr_matrix((np.squeeze(strength_grad), (edges[:,0], edges[:,1])), 
                                 shape=(nnodes, nnodes))
    M_strength_grad_rowSum = M_strength_grad.sum(axis=1)
    Q_grad = (M_strength_grad.multiply(M_strength_rowSum) 
              -M_strength.multiply(M_strength_grad_rowSum)) / np.square(M_strength_rowSum)
    return csr_matrix(Q_grad)


# This is the allclose comparison for sparse matrices
def csr_allclose(a, b, rtol=1e-5, atol = 1e-8):
    c = np.abs(a-b) - rtol*np.abs(b)
    return c.max() <= atol


# This is the allclose comparison for numba.jit
@jit(nopython=True)
def jit_allclose(a, b, rtol=1e-5, atol = 1e-8):
    c = np.abs(a-b) - rtol*np.abs(b)
    return c.max() <= atol


# This function takes a normalized transition matrix Q (n by n), 
# initial state P_init (m by n), and reset probability,
# then use iteration to find the personalized PageRank at convergence.
@jit(nopython=True)
def iterative_PPR(Q, P_init, rst_prob):
    # Q and P_init are already normalized by row sums
    # Takes P_init and a transition matrix to find the PageRank of nodes
    P = P_init.copy()
    rst_prob_P_init = rst_prob*P_init
    P_new =  (1-rst_prob)*np.dot(P, Q) + rst_prob_P_init
    while not(jit_allclose(P, P_new)):
        P = P_new
        P_new =  (1-rst_prob)*np.dot(P, Q) + rst_prob_P_init
    return P_new


# This function takes PPR scores P (m by n), transition matrix Q (n by n), 
# and partial derivatives of Q (Q_grad (n by n)) 
# with respect to one edge feature weight w[l]
# and iteratively calculate the gradient of PPR scores P_grad (m by n)
# with respect to w[l] until converged
# Equation: P = (1-rst_prob)*P.*Q + rst_prob*P_init
# therefore dP/dw = (1-rst_prob) * (dP/dw.*Q+P.*dQ/dw)
def iterative_P_gradient_1feature(P, Q, Q_grad, rst_prob):
    # Initlalize P_grad to be all zeros. See below for P_grad_1_iter
    P_grad = csr_matrix(np.zeros(P.shape))
    P_dot_Qgrad = P.dot(Q_grad)
    P_grad_new = P_grad_1feature_1iter(rst_prob, P_grad, Q, P_dot_Qgrad) 
    # Iteratively calculate P_grad until converged
    while not(csr_allclose(P_grad, P_grad_new)):
        P_grad = P_grad_new
        P_grad_new = P_grad_1feature_1iter(rst_prob, P_grad, Q, P_dot_Qgrad)
    return P_grad_new

# *** This function calculates P_grad for one iteration, 
# which is called by iterative_P_gradient_1feature
def P_grad_1feature_1iter(rst_prob, P_grad, Q, P_dot_Qgrad):
    return (1-rst_prob) * (P_grad.dot(Q)+P_dot_Qgrad)


# This wrapper function first calculates Q_grad, and then P_grad 
# with respect to one edge feature weitht w[l]
# This is called by the following function calc_P_grad_pool 
# using multiprocessing.Pool
def calc_P_grad_1fea(edges, nnodes, M_strength, M_strength_rowSum, Q, P, rst_prob, 
                     strength_grad):
    Q_grad = Q_gradient_1feature(edges, nnodes, M_strength, M_strength_rowSum, 
                                 strength_grad)
    P_grad = iterative_P_gradient_1feature(P, Q, Q_grad, rst_prob)
    return P_grad


# This function calculate P_grad with respect to ALL edge feature weights
# using multiprocessing.Pool
# return a (w by m by n) array
def calc_P_grad_pool(edges, nnodes, M_strength, M_strength_rowSum, Q, P, rst_prob, 
                     strength_grad):
    # Create a partial function of calc_P_grad_1fea, with only one free variabl
    # strength_grad
    calc_P_grad_1fea_partial = functools.partial(calc_P_grad_1fea, edges, nnodes, M_strength, 
                                                 M_strength_rowSum, Q, P, rst_prob)
    # Split strength_grad (e by w) into a list of vectors
    strength_grad_split = np.split(strength_grad.toarray(), strength_grad.shape[1], axis=1)
    
    # For each edge feature weight, calculate P_grad
    n_processes = min(cpu_count(), len(strength_grad_split))
    pool = Pool(processes=n_processes)
    P_grad = pool.map(calc_P_grad_1fea_partial, strength_grad_split)
    pool.close()
    pool.join()
    
    # return a (w by m by n) array
    P_grad = np.array([i.toarray() for i in P_grad])
    return P_grad



# *** Below are two cost functions:

# This is the squared cost function
# Returns sum((pu-pv)**2) / z_uv * y (scalar)
@jit(nopython=True)
def cost_func_squared(P, args):
    cost = 0
    P_dot_PT = np.dot(P, P.T)
    for arg in args:
        (u, v, npairs, y) = arg
        cost += (P_dot_PT[u,u]+P_dot_PT[v,v]-2*P_dot_PT[u,v]) / npairs * y
    return cost

# This is the absolute cost function
# Returns sum(|pu-pv|) / z_uv * y (scalar)
@jit(nopython=True)
def cost_func_abs(P, args):
    cost = 0
    for arg in args:
        (u, v, npairs, y) = arg
        cost += np.sum(np.abs(P[u,:]-P[v,:])) / npairs * y
    return cost



# *** Below are two functions for calculating the gradient of the cost:

# Equation: cost_grad_UV = (pu_grad-pv_grad) .* cost_func_grad
# Dimension: (w by n) .* (n by 1) = (w by 1)

# This is the gradient of the squared cost function
@jit(nopython=True)
def cost_func_gradient_squared(P, P_grad, args):
    cost_grad = np.zeros(P_grad.shape[0])
    # Pgrad_dot_PT = P_grad .* P.T
    # (w by m by n) .* (n by m) = (w by m by m)
    Pgrad_dot_PT = np.zeros((P_grad.shape[0], P_grad.shape[1], 
                             P_grad.shape[1]))
    for l in range(Pgrad_dot_PT.shape[0]):
        Pgrad_dot_PT[l,:,:] = np.dot(P_grad[l,:,:], P.T)
    for arg in args:
        (u, v, npairs, y) = arg
        # cost_grad_uv = (Pu_grad-Pv_grad) .* 2*(Pu.T-Pv.T)
        #              = 2 * (Pu_grad.*Pu.T-Pu_grad.*Pv.T
        #                     -Pv_grad.*Pu.T+Pv_grad.*Pv.T)
        cost_grad += 2*(Pgrad_dot_PT[:,u,u]-Pgrad_dot_PT[:,u,v]
                        -Pgrad_dot_PT[:,v,u]+Pgrad_dot_PT[:,v,v]) / npairs * y
    return cost_grad

# This is the gradient of the absolute cost function
@jit(nopython=True)
def cost_func_gradient_abs(P, P_grad, args):
    cost_grad = np.zeros(P_grad.shape[0])
    for arg in args:
        (u, v, npairs, y) = arg
        # cost_func_grad_uv = sign(pu-pv)
        # (n by 1)
        cost_func_grad = np.sign(P[u,:]-P[v,:])
        cost_grad += np.dot((P_grad[:,u,:]-P_grad[:,v,:]),
                            cost_func_grad) / npairs * y
    return cost_grad



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
def obj_func(P, group_labels, group2npairs, lam, w, loss='squared', 
             norm_type='L2', maximize_diff=False, validation=False):
    if issparse(P):
        P = P.toarray()
    # Compute cost from PPR scores
    nsamples = len(group_labels)
    args=[]
    for u in range(nsamples):
        for v in range(u+1, nsamples):
            if group_labels[u] == group_labels[v]:
                group = group_labels[u]
                y = 1
            else:
                if not maximize_diff:
                    continue
                group = 'diff'
                y = -1
            args.append((u, v, group2npairs[group], y)) 
#    print 'obj_func finished assign args:', time.strftime("%H:%M:%S")

    # The cost is a scalar
    if loss == 'squared':
        # sum((pu-pv)**2) / z_uv * y
        cost = cost_func_squared(P, args)
    elif loss == 'absolute':
        # sum(|pu-pv|) / z_uv * y
        cost = cost_func_abs(P, args)
#    print 'obj_func finished calculating cost:', time.strftime("%H:%M:%S")
        
    # Retrun the cost without norm if it is a validation set
    if validation:
        return cost

    if norm_type == 'L2':
        # Calculate the L2 norm squared of edge feature weights
        norm = np.dot(w, w)
    elif norm_type == 'L1':
        # Calculate the L1 norm of edge feature weights
        norm = np.sum(np.abs(w))

#    print 'obj_func all finished:', time.strftime("%H:%M:%S")
    return lam*norm + cost


# This is the gradient of the objective function
# Equation: J_grad = lam*2*w 
#                    + sum_over_u_and_v(y*(pu_grad-pv_grad).*(2*(pu-pv))/z_uv)
# where y = 1 if u and v belong to the same group, otherwise y = 0 (or -1);
# z_uv is the total number of comparisons in the group where u and v belongs to
# P and P_grad are provided to derive J_grad
# (J_grad is a vector with the length of w)
def obj_func_gradient(P, P_grad, group_labels, group2npairs, lam, w, 
                      loss='squared', norm_type='L2', maximize_diff=False):
    if issparse(P):
        P = P.toarray()
    # Compute the gradient of cost function with respect to edge feature weights w
    nsamples = len(group_labels)
    args = []
    for u in range(nsamples):
        for v in range(u+1, nsamples):
            if group_labels[u] == group_labels[v]:
                group = group_labels[u]
                y = 1
            else:
                if not maximize_diff:
                    continue
                group = 'diff'
                y = -1
            args.append((u, v, group2npairs[group], y))
#    print 'obj_func_gradient finished assign args:', time.strftime("%H:%M:%S")

    # The gradient of the cost is a vector (w by 1) 
    if loss == 'squared':
        cost_grad = cost_func_gradient_squared(P, P_grad, args)
    elif loss == 'absolute':
        cost_grad = cost_func_gradient_abs(P, P_grad, args)
        
    print 'obj_func_gradient finished calculating cost_grad:', time.strftime("%H:%M:%S")

    if norm_type == 'L2':
        # Calculate the gradient of the L2 norm squared term
        norm_grad = 2 * w
    elif norm_type == 'L1':
        # Calculate the gradient of the L1 term
        norm_grad = np.sign(w)
    
#    print 'obj_func_gradient all finished:', time.strftime("%H:%M:%S")
    return lam*norm_grad + cost_grad


# For each group, calculate the number of sample-pairs 
# for normalization in the objective function
def count_numbers_per_group(group_labels):
    group2nsamples = Counter(group_labels)
    group2npairs = {}
    sum_npairs_withingroup = 0
    for group in group2nsamples:
        group2npairs[group] = 0.5 * group2nsamples[group] * (group2nsamples[group]-1)
        sum_npairs_withingroup += group2npairs[group]
    nsamples = len(group_labels)
    group2npairs['diff'] = 0.5*nsamples*(nsamples-1) - sum_npairs_withingroup
    return group2npairs


####################################################################################################
# 3. This part of code uses gradient descent to minimize the objective function
####################################################################################################

# This wrapper function sequentially calculate Q, Q_grad, P, P_grad to derive J and J_grad
def calculate_J_and_gradient(edges, features, nnodes, P_init, rst_prob, group_labels, lam, w, 
                             loss='squared', norm_type='L2', maximize_diff=False, 
                             P_init_validation=None, group_labels_validation=None):
    # Generate transition matrix Q (n by n), edge strength matrix (n by n), its row sums (n by 1),
    # and gradient of edge strength (e by w) according to edge features and weights
    Q, M_strength, M_strength_rowSum, strength_grad = strength_Q_and_gradient(edges, nnodes, 
                                                                              features, w)
    print 'finished calculating strength_grad:', time.strftime("%H:%M:%S")
    # Calculate Personalized PageRank (PPR)
    P = csr_matrix(iterative_PPR(Q.toarray(), renorm(P_init).toarray(), rst_prob))
    print 'network propagation finished:', time.strftime("%H:%M:%S")
    
    # Calculate the gradient of PPR (w by m by n)
    P_grad = calc_P_grad_pool(edges, nnodes, M_strength, M_strength_rowSum, Q, P, rst_prob, 
                              strength_grad)
    print 'finished calculating P_grad using pool:', time.strftime("%H:%M:%S")
    
    # For each group, calculate the number of sample-pairs 
    # to normalize in the objective function
    group2npairs = count_numbers_per_group(group_labels)
    # Calculate objective function J (scalar), 
    # and its gradient J_grad (length w vector)
    J = obj_func(P, group_labels, group2npairs, lam, w, loss, norm_type, maximize_diff)
    J_grad = obj_func_gradient(P, P_grad, group_labels, group2npairs, lam, w, 
                               loss, norm_type, maximize_diff)
    J_valication = ''
    if P_init_validation is not None:
        P_validation = iterative_PPR(Q.toarray(), renorm(P_init_validation).toarray(), rst_prob)
        print 'network propagation finished:', time.strftime("%H:%M:%S")
        group2npairs_validation = count_numbers_per_group(group_labels_validation)
        J_valication = obj_func(P_validation, group_labels_validation, group2npairs_validation, 
                                lam, w, loss, norm_type, maximize_diff, validation=True)
    return J, J_grad, J_valication


# * This is the main function
# This function trains the edge feature weights of Supervised Random Walk
# using gradient descent functions
# Inputs: edges (e by 2), edge features (e by w), number of nodes,
# initial state P_init (m by n), reset probability, group labels (length m vector), 
# regularization parameter labmda, 
# standard deviation of initial edge feature weights, squared or absolute loss,
# L1 or L2 norm, whether to maximize the distance between groups, 
# learning rate, and the function for updating parameters
# (optionally add a validation set to evaluate the performance)
def train_SRW_GD(edges, features, nnodes, P_init, rst_prob, group_labels, lam,
                 w_init_sd=0.01, loss='squared', norm_type='L2', maximize_diff=False, 
                 learning_rate=0.1, update_w_func='Adam', P_init_validation=None,
                 group_labels_validation=None, **kwargs):
    # t is the iteration counter
    t = 0
    # Initialize edge feature weights from a Gaussian distribution, 
    # with standard deviation w_init_sd
    w = np.random.normal(scale = w_init_sd, size = features.shape[1])
    # Calculate the initlal J, and J_grad
    J, J_grad, J_valication = calculate_J_and_gradient(edges, features, nnodes, P_init, 
                                                       rst_prob, group_labels, lam, w, 
                                                       loss, norm_type, maximize_diff, 
                                                       P_init_validation, 
                                                       group_labels_validation)
    print '\n***', t, 'iteration: J is ', J, J_valication
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
    J_new, J_grad, J_valication = calculate_J_and_gradient(edges, features, nnodes, P_init, 
                                                           rst_prob, group_labels, lam, w, 
                                                           loss, norm_type, maximize_diff, 
                                                           P_init_validation, 
                                                           group_labels_validation)
    print '\n***', t, 'iteration: J is ', J_new, J_valication
    
    while not np.allclose(J, J_new):
        t += 1
        J = J_new
        # Update w
        update_w(update_w_func, w, J_grad, learning_rate, m, n, t, PI_mu_t, v, **kwargs)
        # Update J and J_grad
        J_new, J_grad, J_valication = calculate_J_and_gradient(edges, features, nnodes, P_init, 
                                                               rst_prob, group_labels, lam, w, 
                                                               loss, norm_type, maximize_diff, 
                                                               P_init_validation, 
                                                               group_labels_validation)
        print '\n***', t, 'iteration: J is ', J_new, J_valication
    
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
#    print 'finished updating w:', time.strftime("%H:%M:%S")



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
# 3. This part of code uses L-FGBS to minimize the objective function
####################################################################################################

# This wrapper function sequentially calculate Q and P to derive J
def calculate_J(edges, features, nnodes, P_init, rst_prob, group_labels, group2npairs, lam, w, 
                loss='squared', norm_type='L2', maximize_diff=False):
    # Generate transition matrix Q (n by n) according to edge features and weights
    Q = generate_Q(edges, nnodes, features, w)
    # Calculate Personalized PageRank (PPR)
    P = iterative_PPR(Q.toarray(), renorm(P_init).toarray(), rst_prob)
    # Calculate objective function J (scalar)
    J = obj_func(P, group_labels, group2npairs, lam, w, loss, norm_type, maximize_diff)
    return J


# This wrapper function sequentially calculate Q, Q_grad, P, P_grad to derive J and J_grad
def calculate_J_gradient(edges, features, nnodes, P_init, rst_prob, group_labels, group2npairs, 
                         lam, w, loss='squared', norm_type='L2', maximize_diff=False):
    # Generate transition matrix Q (n by n), edge strength matrix (n by n), its row sums (n by 1),
    # and gradient of edge strength (e by w) according to edge features and weights
    Q, M_strength, M_strength_rowSum, strength_grad = strength_Q_and_gradient(edges, nnodes, 
                                                                              features, w)
    # Calculate Personalized PageRank (PPR)
    P = csr_matrix(iterative_PPR(Q.toarray(), renorm(P_init).toarray(), rst_prob))
    
    # Calculate the gradient of PPR (w by m by n)
    P_grad = calc_P_grad_pool(edges, nnodes, M_strength, M_strength_rowSum, Q, P, rst_prob, 
                              strength_grad)
    
    # Calculate the gradient of the objective function J_grad (length w vector)
    J_grad = obj_func_gradient(P, P_grad, group_labels, group2npairs, lam, w, loss, 
                               norm_type, maximize_diff)
    return J_grad


# * This is the main function
# This function trains the edge feature weights of Supervised Random Walk using L-BFGS
# Inputs: edges (e by 2), edge features (e by w), number of nodes,
# initial state P_init (m by n), reset probability, group labels (length m vector), 
# regularization parameter labmda, standard deviation of initial edge feature weights, 
# squared or absolute loss, L1 or L2 norm, and whether to maximize the distance between groups, 
def train_SRW_BFGS(edges, features, nnodes, P_init, rst_prob, group_labels, lam, w_init_sd=0.01, 
                   loss='squared', norm_type='L2', maximize_diff=False, P_init_validation=None,
                   group_labels_validation=None):
    # Initialize edge feature weights from a Gaussian distribution, 
    # with standard deviation w_init_sd
    w_init = np.random.normal(scale = w_init_sd, size = features.shape[1])
    # For each group, calculate the number of sample-pairs 
    # to normalize in the objective function
    group2npairs = count_numbers_per_group(group_labels)
    
    # scipy's L-BFGS-B optimizer is called to iteratively optimize J
    # J and J_grad are the main input to the BFGS optimizer
    # functools.partial is used to generate partial functions of J and J_grad 
    # with only one free argument w
    w, J, d = fmin_l_bfgs_b(functools.partial(calculate_J, edges, features, nnodes, P_init, 
                                              rst_prob, group_labels, group2npairs, lam, 
                                              loss=loss, norm_type=norm_type, 
                                              maximize_diff=maximize_diff), w_init,
                            fprime = functools.partial(calculate_J_gradient, edges, features, 
                                                       nnodes, P_init, rst_prob, group_labels, 
                                                       group2npairs, lam, loss=loss, 
                                                       norm_type=norm_type, 
                                                       maximize_diff=maximize_diff))
    
    J_validation = ''
    if P_init_validation is not None:
        Q = generate_Q(edges, nnodes, features, w)
        P_validation = iterative_PPR(Q.toarray(), renorm(P_init_validation).toarray(), rst_prob)
        group2npairs_validation = count_numbers_per_group(group_labels_validation)
        J_validation = obj_func(P_validation, group_labels_validation, group2npairs_validation, 
                                lam, w, loss, norm_type, maximize_diff, validation=True)
    
    return w, J, d, J_validation


