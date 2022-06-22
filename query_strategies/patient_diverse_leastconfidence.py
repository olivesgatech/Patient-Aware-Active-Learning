import numpy as np
import random
from .sampler import Sampler


class PatientDiverseLeastConfidenceSampler(Sampler):
    '''Class for sampling the highest gradnorm. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs):
        '''Constructor implemented in sampler'''
        super(PatientDiverseLeastConfidenceSampler, self).__init__(n_pool, start_idxs)

    def query(self, n, probs):
        '''Returns the least confident prediction probabilities.
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
        probabilities = probs['probs']

        inds = []
        for id in unique_IDs:
            # get max probs
            probs = np.max(probabilities[id_idx[id]], axis=1)
            prob_inds = np.argmax(probs)
            index = id_idx[id][prob_inds]
            inds.append(index)
        return inds



