import numpy as np
import random
from .sampler import Sampler

class PatientDiverseSampler(Sampler):
    '''Class for randomly sampling the the most diverse set of patient in each batch. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs):
        '''Constructor implemented in sampler'''
        super(PatientDiverseSampler, self).__init__(n_pool, start_idxs)

    def query(self, n, patient_dict):
        # indices for patients with same ID
        id_idx = dict()
        for idx, id in enumerate(patient_dict['IDs']):
            if id not in id_idx:
                id_idx[id] = [idx]
            else:
                id_idx[id].append(idx)

        # Random list of unique IDs
        unique_IDs = random.choices(list(id_idx.keys()), k=n)

        # select a random idx for each ID
        inds = []
        for id in unique_IDs:
            rand_idx = random.choice(id_idx[id])
            inds.append(rand_idx)

        return inds

