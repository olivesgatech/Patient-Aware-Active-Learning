import numpy as np
import random
from .sampler import Sampler


class PatientDiverseEntropySampler(Sampler):
    '''Class for sampling the highest gradnorm. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs):
        '''Constructor implemented in sampler'''
        super(PatientDiverseEntropySampler, self).__init__(n_pool, start_idxs)

    def query(self, n, probs):
        '''Returns samples with highest entropy in the output distribution.
        Parameters:
            :param probs: datastructure containing the sigmoid probabilities and the index list
            :type probs: dict
            :param n: number of samples to be queried
            :type n: int'''
        patients = probs['IDs']
        id_idx = dict()
        for idx, id in enumerate(patients):
            if id not in id_idx:
                id_idx[id] = [idx]
            else:
                id_idx[id].append(idx)

        # Random list of unique IDs
        unique_IDs = random.choices(list(id_idx.keys()), k=n)

        # get probabilities and their indices
        probabilities = probs['probs']

        # select idx with highest entropy for each ID
        inds = []
        for id in unique_IDs:
            logs = np.log2(probabilities[id_idx[id]])
            mult = logs*probabilities[id_idx[id]]
            entropy = np.sum(mult, axis=1)
            prob_inds = np.argmax(entropy)
            index = id_idx[id][prob_inds]
            inds.append(index)
        return inds
