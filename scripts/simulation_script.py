import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score

import numpy as np
import pandas as pd
from datetime import datetime

from collections import Counter
import itertools

from math import comb, ceil
from scipy import stats
import random
import string

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

from tqdm import tqdm

import multiprocessing as mp

import pickle

import sys
sys.path.insert(1, '../scripts')
import modeling_v3

def network_out_of_relationships(rel):
    g = nx.Graph()

    for p1, v in rel.items():
        g.add_node(p1)
        for p2, s in v.items():
            if s!=0:
                g.add_edge(p1,p2)
    
    return g

def connected_components(rel):
    g = network_out_of_relationships(rel)
    
    return [len(cc) for cc in nx.connected_components(g)]


def multi_agent_simulation(random_seed = 89, n_beliefs = 6, m = 15, gamma = .7, n_agents = 2, last_step = 10000, sigma = .3, times_more_likely = 1):
    
    np.random.seed(random_seed)
    random.seed(random_seed)

    # initialize the belief network
    bn = modeling_v3.create_belief_network(m = m,
                                            n_beliefs = n_beliefs,
                                            complete = True)

    # initialize the agent
    agents = modeling_v3.init_agents(bn, n_agents)

    propositions = [*bn.nodes()]
    prop_combs = [*itertools.combinations(propositions, 2)]
    agent_list = [*agents.keys()]

    ratios = np.array([times_more_likely] + [1]*5)
    probs = ratios/sum(ratios)

    simulation_steps = [*range(last_step+1)]
    track = {}

    for t in tqdm(simulation_steps):

        i, j = np.random.choice(agent_list, 2, replace=False) # choose an agent to be the sender and the other one will be the receiver

        # choose a message with biased probabilities to convey
        b1 = np.random.choice(propositions, 1, p=probs)[0]

        # the second belief depends on the relationship between belief 1 and 2
        relationships, likelihoods = modeling_v3.determine_relationships_likelihoods(agents[i]['association_matrix'], gamma)
        
        choices = [k for k,v in relationships[b1].items() if v != 0]

        if len(choices) == 0: # if the node is irrelevant to the others, choose one randomly
            temp = propositions.copy()
            temp.remove(b1)
            b2 = np.random.choice(temp)
        
        else: # else, choose one among relevants
            b2 = np.random.choice(choices)

        beliefs_to_convey = [b1,b2]
        
        message_to_convey = modeling_v3.deduce_message_to_convey(beliefs_to_convey, agents[i]['beliefs']) # deduce the message to convey

        # update the agent's association matrix accordingly
        modeling_v3.update_association_matrix(message_to_convey, agents[j]['association_matrix'])
        
        # assess the relationships and likelihoods of two propositions being exhibited together
        relationships, likelihoods = modeling_v3.determine_relationships_likelihoods(agents[j]['association_matrix'], gamma)

        # update agent's belief randomly
        beliefs_new = modeling_v3.update_belief(message_to_convey['e1'], agents[j]['beliefs'], sigma)
        
        # extract the relationship
        relationship = relationships[b1][b2]

        if relationship != 0:

            energy_t = modeling_v3.energy(message_to_convey = message_to_convey,
                                        beliefs = agents[j]['beliefs'],
                                        relationship = relationship)
            
            energy_t_plus_1 = modeling_v3.energy(message_to_convey = message_to_convey,
                                                beliefs = beliefs_new,
                                                relationship = relationship)

            if energy_t_plus_1 < energy_t:
                agents[j]['beliefs'] = beliefs_new
        
        if t % 200 == 0:
            track[t] = {}
            beliefs = np.array([[*a['beliefs'].values()] for a in agents.values()]).copy()
            track[t]['beliefs'] = beliefs

            if n_agents >= 4:
                # Number of clusters
                cluster_count, silhouette = modeling_v3.optimal_clustering(beliefs, 20)
                track[t]['cluster_count'] = cluster_count
                track[t]['silhouette_score'] = silhouette

            # Internal energies
            internal_energies = modeling_v3.total_energy(agents, gamma)
            track[t]['internal_energies'] = internal_energies

            # Belief similarity
            pairwise_belief_dist = modeling_v3.belief_distances(agents)
            track[t]['pairwise_belief_dist'] = pairwise_belief_dist

            # Compute the interpretative distance between agents
            likelihoods_all_agents = {}
            for agent, v in agents.items():
                r, l = modeling_v3.determine_relationships_likelihoods(v['association_matrix'], gamma)
                likelihoods_all_agents[agent] = l
            
            interpretative_dist = interpretative_agreement(likelihoods_all_agents, prop_combs)
            track[t]['interpretative_dist'] = interpretative_dist
            track[t]['likelihoods_all_agents'] = likelihoods_all_agents
            
    return track

def interpretative_agreement(likelihoods_all_agents, prop_combs):
    
    likelihood_piles = []

    for agent, v in likelihoods_all_agents.items():
        _ = []
        for p1, p2 in prop_combs:
            _.append(v[p1][p2])
        
        likelihood_piles.append(_)

    pairwise_dist = pairwise_distances(np.array(likelihood_piles), metric='manhattan')
    
    return np.mean(pairwise_dist[np.triu_indices_from(pairwise_dist, k=1)])

import warnings
warnings.filterwarnings('ignore')


n_beliefs = 6
m = 15
gamma_vals = [0, .6, .7, .8]
n_agents = 100
last_step = 100000
sigma = .3
times_more_likely = 1
repetition = 100

track = {}

for gamma in gamma_vals:

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    print(dt_string, gamma)
    track[gamma] = {}

    random_states = [np.random.randint(0, 100000) for i in range(repetition)]    
    
    params = [*zip(random_states,
                    [n_beliefs] * repetition, 
                    [m] * repetition,
                    [gamma] * repetition,
                    [n_agents] * repetition,
                    [last_step] * repetition,
                    [sigma] * repetition,
                    [times_more_likely] * repetition)]

    pool = mp.Pool(processes=25)
    temp = pool.starmap(multi_agent_simulation, params, chunksize=1)
    pool.close()

    track[gamma] = temp
    del temp

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
with open(f'../data/results_multi_agent_{dt_string}.pkl', 'wb') as f:
    pickle.dump(track, f)