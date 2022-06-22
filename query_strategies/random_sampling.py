import numpy as np
from .sampler import Sampler
import pdb


class RandomSampling(Sampler):
    '''Class for random sampling algorithm. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs):
        super(RandomSampling, self).__init__(n_pool, start_idxs)

    def query(self, n):
        '''Performs random query of points'''
        inds = np.where(self.total_pool == 0)[0]
        return inds[np.random.permutation(len(inds))][:n]