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

import sys
sys.path.insert(1, '../scripts')
import modeling

# prepare the model parameters
max_nodes_to_connect = 6
repetition = 50
n_agents = 20
last_step = 500000

n_nodes_to_connect = list(np.array([[i]*repetition for i in range(1,max_nodes_to_connect+1)]).flatten())
simulation_count = repetition * len(np.unique(n_nodes_to_connect))
params = [*zip([*range(simulation_count)], n_nodes_to_connect, [n_agents] * simulation_count, [last_step] * simulation_count)]

# fire up the CPUs and run the simulations parallelly 
pool = mp.Pool(processes=32)
results = pool.starmap(modeling.simulate_multiple_agents, params, chunksize=1)
pool.close()

# save
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
with open(f'../data/results_{dt_string}.pkl', 'wb') as f:
    pickle.dump(results, f)