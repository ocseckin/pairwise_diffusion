import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score

import numpy as np
import pandas as pd
from datetime import datetime

from collections import Counter
import itertools

from math import comb, ceil
import random
import string

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import multiprocessing as mp

import pickle


def create_belief_network(m = 15, n_beliefs = 6, complete = True):
    
    if complete:
        m = comb(n_beliefs, 2)

    is_connected = False # initialize the belief network
    while is_connected == False:
        bn1 = nx.gnm_random_graph(n=n_beliefs, m = m)
        if nx.number_of_selfloops(bn1) == 0:
            is_connected = nx.is_connected(bn1)

    propositions = list(string.ascii_uppercase[:n_beliefs]) # get the proposition names from the alphabet

    bn1 = nx.relabel_nodes(bn1, {n:propositions[n] for n in bn1.nodes()}) # relabel the nodes with letters from the alphabet

    return bn1


def compose_networks(bn1, bn2, n_nodes_to_connect = 1, random=False):
    
    edges_to_add = []

    n1 = np.random.choice([*bn1.nodes()])

    for c in range(n_nodes_to_connect):
        pair_exists = True
        while pair_exists:
            if random == False:
                pair = (n1, np.random.choice([*bn2.nodes()]))
            else:
                pair = (np.random.choice([*bn1.nodes()]), np.random.choice([*bn2.nodes()]))
            if pair not in edges_to_add:
                pair_exists = False

        edges_to_add.append(pair)

    bn_composed = nx.compose(bn1,bn2)
    propositions = [*bn_composed.nodes()]

    for edge in edges_to_add:
        bn_composed.add_edge(edge[0], edge[1])

    return bn_composed


def flatten(xss):
    return [x for xs in xss for x in xs]


def init_association_matrix(belief_network, propositions = ['A','B','C','D']):
    
    propositions_w_sign = ["+"+p for p in propositions] + ["-"+p for p in propositions] # add the signs

    propositions_w_sign = sorted(propositions_w_sign, key=lambda x: x[1]) # sort by proposition, not sign

    association_matrix = pd.DataFrame(index=propositions_w_sign,  # create the association matrix
                                      columns=propositions_w_sign).fillna(0)

    for e in belief_network.edges(): # fill up 
        for s in [*itertools.product(['+','-'],['+','-'])]:
            association_matrix[f'{s[0]}{e[0]}'][f'{s[1]}{e[1]}'] = 1
            association_matrix[f'{s[0]}{e[1]}'][f'{s[1]}{e[0]}'] = 1

    return association_matrix


def init_beliefs(propositions):
    
    beliefs = np.random.uniform(-1, 1, size=len(propositions))
    beliefs = {e:b for e,b in zip(sorted(propositions), beliefs)}
    
    return beliefs


def bayesian(e1, e2, association_matrix):
    
    evidence = association_matrix[f'-{e2[1]}'].sum() + association_matrix[f'+{e2[1]}'].sum()
    joint_1 = association_matrix[f'+{e2[1]}'][f'+{e1[1]}'] + association_matrix[f'-{e2[1]}'][f'-{e1[1]}']
    joint_2 = association_matrix[f'-{e2[1]}'][f'+{e1[1]}'] + association_matrix[f'+{e2[1]}'][f'-{e1[1]}']
    
    likelihood_pos_rel = joint_1/evidence
    likelihood_neg_rel = joint_2/evidence
    
    return likelihood_pos_rel, likelihood_neg_rel


def find_irrelevant_propositions(association_matrix, gamma = .1):

    # first, compress the matrix to remove signs in front of the propositions
    compressed_matrix = association_matrix.groupby(lambda x: x[1], axis=1).sum().groupby(lambda x: x[1], axis=0).sum()

    # normalize the association matrix with the total number of associations
    normalized_association_matrix = compressed_matrix / compressed_matrix.sum().sum()

    # transform into a numpy array
    normalized_association_matrix = normalized_association_matrix.to_numpy()

    # flatten
    flattened = normalized_association_matrix.flatten()

    # find non-zero values (-A,+A or +A,+A etc. shouldn't be counted)
    non_zero_values = flattened[np.where(flattened!=0)]

    # determine the threshold for relevant propositions
    relevance_threshold = np.percentile(non_zero_values, gamma*100, method='lower')

    # find the spots on the matrix that exceed the relevance threshrold
    i, j = np.where(normalized_association_matrix < relevance_threshold)

    propositions = np.array([*compressed_matrix.columns])

    irrelevant_propositions = set(["".join(sorted(p)) for p in propositions[[*zip(i,j)]]])

    return irrelevant_propositions


def determine_relationships_likelihoods(association_matrix, gamma):

    propositions = sorted(list(set([c[1] for c in association_matrix.columns])))
    pairs = [*itertools.permutations(propositions,2)]

    relationships = {p1:{p2:None for p2 in propositions if p1!=p2} for p1 in propositions}
    likelihoods = {p1:{p2:None for p2 in propositions if p1!=p2} for p1 in propositions}
    joint_probs = {}

    for pair in pairs:
        b1 = pair[0]
        b2 = pair[1]
        
        evidence = association_matrix[f'-{b1}'].sum() + association_matrix[f'+{b1}'].sum()

        joint_1 = association_matrix[f'+{b1}'][f'+{b2}'] + association_matrix[f'-{b1}'][f'-{b2}']
        joint_2 = association_matrix[f'-{b1}'][f'+{b2}'] + association_matrix[f'+{b1}'][f'-{b2}']
        
        if joint_1 > joint_2:
            relationships[b1][b2] = 1
        else:
            relationships[b1][b2] = -1

        #likelihoods[b1][b2] = (joint_1+joint_2)/evidence

    irrelevant_propositions = find_irrelevant_propositions(association_matrix, gamma)
    for pair in irrelevant_propositions:
        relationships[pair[0]][pair[1]] = 0
        relationships[pair[1]][pair[0]] = 0

    return relationships, likelihoods


def energy(message_to_convey, beliefs, relationship):
    
    e1 = message_to_convey['e1']
    e2 = message_to_convey['e2']
    
    return - beliefs[e1[1]]*beliefs[e2[1]]*relationship


def energy_comprehensive(message_to_convey, beliefs, association_matrix, belief_network, gamma):
    
    energy_sum = 0

    e2 = message_to_convey['e2']
    neighbors = [*belief_network.neighbors(e2[1])]

    for e1 in neighbors:

        relationship = infer_relationship(message_to_convey = message_to_convey, 
                                          association_matrix = association_matrix,
                                          gamma = gamma)

        e = energy(message_to_convey = message_to_convey, beliefs = beliefs, relationship = relationship)

        denominator = association_matrix[f"-{e2[1]}"].sum() + association_matrix[f"+{e2[1]}"].sum()
        nominator = association_matrix[f"-{e2[1]}"][f"-{e1}"].sum() + association_matrix[f"+{e2[1]}"][f"-{e1}"].sum() + association_matrix[f"-{e2[1]}"][f"+{e1}"].sum() + association_matrix[f"+{e2[1]}"][f"+{e1}"].sum()
        weight = nominator / denominator

        energy_sum += e * weight

    return energy_sum


def total_energy(agents, gamma):

    energies = []

    for a, v in agents.items():
        
        beliefs = v['beliefs']
        association_matrix = v['association_matrix']
        propositions = [*beliefs.keys()]
        pairs = [*itertools.permutations(propositions,2)]

        relationships, likelihoods = determine_relationships_likelihoods(association_matrix=association_matrix, gamma=gamma)
        internal_energy = 0
        normalizer = 0
        
        for b1, b2 in pairs:
            internal_energy += beliefs[b1] * beliefs[b2] * relationships[b1][b2]
            
            if relationships[b1][b2] != 0:
                normalizer += 1

        energies.append(- internal_energy / normalizer)

    return energies


def update_belief(e1, beliefs, sigma=1):
    
    update_term = np.random.normal(0, sigma) # get an update term randomly
    
    beliefs_new = beliefs.copy() # make a copy of beliefs
    
    beliefs_new[e1[1]] = beliefs_new[e1[1]] + update_term # update the belief
    
    if beliefs_new[e1[1]] > 1:
        beliefs_new[e1[1]] = 1
        
    if beliefs_new[e1[1]] < -1:
        beliefs_new[e1[1]] = -1
    
    return beliefs_new


def update_association_matrix(message_to_convey, association_matrix):
    
    e1 = message_to_convey['e1']
    e2 = message_to_convey['e2']
    
    association_matrix[e2][e1] += 1
    association_matrix[e1][e2] += 1


def init_agents(belief_network, n_agents = 2):
    
    propositions = [*belief_network.nodes()]
    
    agents = {}
    
    for a in range(n_agents):
        agents[a] = {}
        agents[a]['association_matrix'] = init_association_matrix(belief_network, propositions)
        agents[a]['beliefs'] = init_beliefs(propositions)
    
    return agents


def deduce_message_to_convey(beliefs_to_convey, agent_beliefs):
    
    message_to_convey = {}
    
    for i, b in enumerate(beliefs_to_convey):
        if np.sign(agent_beliefs[beliefs_to_convey[i]]) == -1:
            message_to_convey[f"e{i+1}"] = f"-{beliefs_to_convey[i]}"
        else:
            message_to_convey[f"e{i+1}"] = f"+{beliefs_to_convey[i]}"
    
    return message_to_convey


def optimal_clustering(beliefs, n_agents):
    
    silhouettes = []
    n_clusters = [*range(2,n_agents)]

    for k in n_clusters:
        clustering = KMeans(n_clusters=k, n_init = 'auto').fit(beliefs)
        silhouettes.append(silhouette_score(beliefs, clustering.labels_))
    
    return n_clusters[np.argmax(silhouettes)], silhouettes[np.argmax(silhouettes)]


def evaluatory_agreement(beliefs):
    
    pref_corr = pd.DataFrame(beliefs).T.corr()
    pref_corr = pref_corr.to_numpy()
    pref_corr = pref_corr[np.triu_indices_from(pref_corr, k=1)]
    
    return pref_corr


def interpretative_agreement(propositions, agents):
    
    proposition_combinations = [*itertools.combinations(propositions, 2)]

    posteriors_all = []
    for matrix in [v['association_matrix'] for v in agents.values()]:
        posteriors = []
        for p1, p2 in proposition_combinations:
            posterior_pos, posterior_neg = bayesian(f'+{p1}', f'+{p2}', matrix)
            posteriors.append(posterior_pos)
            posteriors.append(posterior_neg)

        posteriors_all.append(posteriors)

    interpretative_dist = pairwise_distances(np.array(posteriors_all), metric='euclidean')

    interpretative_dist = interpretative_dist[np.triu_indices_from(interpretative_dist, k=1)]
    
    return interpretative_dist


def belief_similarities(agents):

    belief_matrix = np.array([[*a['beliefs'].values()] for a in agents.values()])

    # compute pairwise belief distances
    pairwise_belief_dist = pairwise_distances(belief_matrix, metric='cosine')

    # drop duplicate values
    pairwise_belief_dist = pairwise_belief_dist[np.triu_indices_from(pairwise_belief_dist, k=1)]

    # compute similarity
    pairwise_belief_sim = 1 - pairwise_belief_dist

    return pairwise_belief_sim


def simulate_multiple_agents(sim_no, random_seed, n_beliefs, m, gamma=.1, n_nodes_to_connect=1, n_agents = 2, last_step = 10000, sigma=1, is_composed=False):

    np.random.seed(random_seed)
    random.seed(random_seed)

    bn1 = create_belief_network(m = m,
                                n_beliefs = n_beliefs,
                                complete = True)

    if is_composed:
        bn2 = bn1.copy()
        bn2 = nx.relabel_nodes(bn2,{l1:l2 for l1,l2 in zip([*bn1.nodes],string.ascii_uppercase[len(bn1):len(bn1)+len(bn2)])})
        bn_composed = compose_networks(bn1=bn1, bn2=bn2, n_nodes_to_connect = n_nodes_to_connect, random=False)
    else:
        bn_composed = bn1


    propositions = [*bn_composed.nodes()]
    agents = init_agents(bn_composed, n_agents)
    agent_list = [*agents.keys()]
    node_list = [*bn_composed.nodes()]
    edge_list = [*bn_composed.edges()]

    simulation_steps = [*range(last_step+1)]

    track = {}

    for t in tqdm(simulation_steps):
        
        i,j = np.random.choice(agent_list, 2, replace=False) # choose an agent to be the sender and the other one will be the receiver
        
        # choose the first belief to send
        b1 = np.random.choice(node_list)
        # the second belief depends on the relationship between belief 1 and 2, if unrelated, don't take it
        relationships, likelihoods = determine_relationships_likelihoods(agents[i]['association_matrix'], gamma)
        
        choices = [k for k,v in relationships[b1].items() if v != 0]

        if len(choices) == 0: # if it seems like the node is irrelevant to the others, choose one randomly
            temp = node_list.copy()
            temp.remove(b1)
            b2 = np.random.choice(temp)
        
        else: # else, choose one among relevants
            b2 = np.random.choice(choices)

        beliefs_to_convey = [b1,b2]
        
        message_to_convey = deduce_message_to_convey(beliefs_to_convey, agents[i]['beliefs']) # deduce the message to convey
        update_association_matrix(message_to_convey, agents[j]['association_matrix']) # update receiver's association matrix
        
        relationships, likelihoods = determine_relationships_likelihoods(agents[j]['association_matrix'], gamma)
        
        beliefs_new = update_belief(message_to_convey['e1'], agents[j]['beliefs'], sigma)
        
        relationship = relationships[b1][b2]

        if relationship != 0:

            energy_t = energy(message_to_convey = message_to_convey,
                            beliefs = agents[j]['beliefs'],
                            relationship = relationship)
            
            energy_t_plus_1 = energy(message_to_convey = message_to_convey,
                            beliefs = beliefs_new,
                            relationship = relationship)

            if energy_t_plus_1 < energy_t:
                agents[j]['beliefs'] = beliefs_new


        # keep track of the evolution of the agents
        if t % 50 == 0:
            track[t] = {}
            
            beliefs = np.array([[*a['beliefs'].values()] for a in agents.values()])
            
            # Count of unique belief networks
            unique_belief_networks = np.unique(beliefs,axis=0)
            track[t]['count_unique_belief_networks'] = len(unique_belief_networks)
            
            if n_agents >= 4:

                # Number of clusters
                cluster_count, silhouette = optimal_clustering(beliefs, n_agents)
                track[t]['cluster_count'] = cluster_count
                track[t]['silhouette_score'] = silhouette
                
            # Preference congruence and similarity
            pairwise_belief_sim = belief_similarities(agents)
            track[t]['pairwise_belief_sim'] = pairwise_belief_sim
            
            # Computing evaluatory agreement
            interpretative_dist = interpretative_agreement(propositions, agents)
            track[t]['interpretative_dist'] = np.mean(interpretative_dist)

            # Internal energies
            internal_energies = total_energy(agents, gamma)
            track[t]['internal_energies'] = internal_energies

            # Stop if everybody has either -1 or +1 only
            #if len(np.unique(unique_belief_networks))==2:
            #    track[t]['belief_networks'] = beliefs
            #    break
            if t == last_step:
                track[t]['belief_networks'] = beliefs
                track[t]['association_matrix'] = {k:v['association_matrix'].copy() for k,v in agents.items()}
    
    return {'sim_no':sim_no, 'n_nodes_to_connect':n_nodes_to_connect, 'track':track}