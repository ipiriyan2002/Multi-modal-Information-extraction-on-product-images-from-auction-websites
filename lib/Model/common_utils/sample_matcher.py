#Import torch related packages
import torch


class SampleMatcher:

    def __init__(self, max_samples, sample_ratio, keep_inds=True):
        self.max_samples = max_samples
        self.sample_ratio = sample_ratio
        self.keep_inds = keep_inds


    def getMinMaxSampleSize(self, pos_indexes, neg_indexes):

        #Maximum possible positive sample
        maximum_pos_samples = self.max_samples * self.sample_ratio

        #Number of positive indexes to sample
        num_pos = min(pos_indexes.numel(), maximum_pos_samples)

        #Maximum possible negative samples
        maximum_neg_samples = self.max_samples - num_pos

        #Number of negative indexes to sample
        num_neg = min(neg_indexes.numel(), maximum_neg_samples)

        return num_pos, num_neg

    def __call__(self,pos_indexes, neg_indexes):
        #Get the parameters device and number of positive and negative samples to sample
        device = pos_indexes.device
        num_pos, num_neg = self.getMinMaxSampleSize(pos_indexes, neg_indexes)
        num_pos, num_neg = int(num_pos), int(num_neg)

        """
        Deal with three cases during sampling
        1) More than 0 positive and negative samples
        2) More than 0 positive samples but no negative samples available
        3) More than 0 negative samples but no positive samples available (In this case the positive samples are padded as negative samples)
        DEFAULT CASE: activates mostly when there are no positive and negative samples to work with, hence throws an error
        """
        if num_pos > 0 and num_neg > 0:
            #Using torch randperm to randomly sample a permutation of indexes for both positive and negative
            pos_rand = torch.randperm(pos_indexes.numel(), device=device)
            neg_rand = torch.randperm(neg_indexes.numel(), device=device)

            #Using slicing to get the indexes to keep / remove
            #As this class is abstract and does not need to know what the indexes are used for
            #It return either the keep indexes or remove indexes depending on user need
            sampled_pos = pos_rand[num_pos:] if not(self.keep_inds) else pos_rand[:num_pos]
            sample_neg = neg_rand[num_neg:] if not(self.keep_inds) else neg_rand[:num_neg]

            #Sample the indexes
            pos_samples = pos_indexes[sampled_pos]
            neg_samples = neg_indexes[sample_neg]

            #Get a single torch tensor of indexes
            final_idxs = torch.cat([pos_samples, neg_samples])

        elif num_pos > 0 and num_neg == 0:
            #Same as before but only for positive samples
            pos_rand = torch.randperm(pos_indexes.numel(), device=device)
            sampled_pos = pos_rand[num_pos:] if not(self.keep_inds) else pos_rand[:num_pos]
            final_idxs = pos_indexes[sampled_pos]

        elif num_neg > 0 and num_pos == 0:
            #Same as before but only for negative samples
            neg_rand = torch.randperm(neg_indexes.numel(), device=device)
            sample_neg = neg_rand[num_neg:] if not(self.keep_inds) else neg_rand[:num_neg]
            final_idxs = neg_indexes[sample_neg]
        else:
            raise ValueError("Cannot have no positive and negative boxes")

        return final_idxs, num_pos, num_neg




        



