#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on 09/16/22

@author: Jorge Iravedra
'''

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.interpolate import interp1d
from planar import BoundingBox, Polygon
import h5py
import difflib
import pickle
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from tqdm import tqdm

### pre-processing functions

def smooth(feat, win=5):
	return gaussian_filter1d(feat, 11)

def delete_outliers(a, st = 4):

	a[np.abs(a - a.mean()) > st * a.std()] = np.nan

	return a


def fill_missing(Y, kind='cubic'):
    from scipy.interpolate import interp1d
    initial_shape = Y.shape
    Y = Y.reshape((initial_shape[0], -1))
    for i in range(Y.shape[-1]):
        y = Y[:,i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y

def collated_zscore(a, u, s):
	
	return (a - u)/s

def feat_process(a, remove_outlier=False, filt=False, normal_zscoring=False, collated_min=False, collated_max=False, collated_zscoring=False):

	# remove outliers --> gaussian filter of data --> interpolate data --> rescale data
	
	processed_feat = a.copy()
	
	if remove_outlier:

		processed_feat = delete_outliers(processed_feat, st=3)
		
	if filt:

		processed_feat = smooth(processed_feat, win=9)
	
	if remove_outlier:
		
		processed_feat = fill_missing(processed_feat)
		
	if collated_zscoring: 
		
		processed_feat = collated_zscore(processed_feat, collated_min, collated_max)
		
	if normal_zscoring:
		
		processed_feat = zscore(processed_feat)

	return processed_feat

def save_session_feats(feats, path):

	with open(path, 'wb') as handle:
		pickle.dump(feats, handle, protocol=pickle.HIGHEST_PROTOCOL)


###

feats_path = '/scratch/gpfs/jmig/feature_processing/extracted_features/'
save_path = '/scratch/gpfs/jmig/feature_processing/processed_features_not4training/'

feats = np.sort([feats_path + x for x in os.listdir(feats_path) if '.pickle' in x])
nfeatures = len(pd.read_pickle(feats[0]).columns)
print('Working with %i features across %i videos...' % (nfeatures, len(feats)))

# collate features

print('Collating features...')

collated_means = np.zeros((len(feats), nfeatures))
collated_stds = collated_means.copy()

# Initialize tqdm for the loading bar
with tqdm(total=len(feats)) as pbar:
    for i, feat in enumerate(feats):
        collated_means[i, :] = pd.read_pickle(feat).mean(axis=0).values
        collated_stds[i, :] = pd.read_pickle(feat).std(axis=0).values
        # Update the loading bar
        pbar.update(1)

# Calculate collated_mean and collated_std
collated_mean = collated_means.mean(axis=0)
collated_std = collated_stds.mean(axis=0)

# Create a DataFrame for z-score metrics
zscore_metrics = pd.DataFrame(np.vstack([collated_mean, collated_std]))
zscore_metrics.columns = pd.read_pickle(feats[0]).columns

# process features (z-scoring)

px2cm = 25.0142
feats2process = [x for x in feats if '927R' in x or '927L' in x or '933R' in x or '583L2' in x or '583B' in x] # modify here if need to process subsample of feats
print('Beginning feature processing')
for feat in feats2process:

	print('Processing %s' % feat)
	
	features = pd.read_pickle(feat)

	proc_feats = pd.DataFrame()
	
	for col in features.columns:
		
		if 'roc' in col or 'proximity' in col or 'head-' in col or 'nose-' in col or 'head2' in col or 'tti2' in col or 'tailbase2' in col or 'trunk2' in col and 'angle' not in col:
			proc_feats[col] = feat_process(features[col].values/px2cm, 
										   filt=True,
										   remove_outlier=False,
										   collated_zscoring=True,
										   collated_min=zscore_metrics[col][0]/px2cm,
										   collated_max=zscore_metrics[col][1]/px2cm)

		else:
			proc_feats[col] = feat_process(features[col].values, 
										   filt=True, 
										   remove_outlier=False,
										   collated_zscoring=True,
										   collated_min=zscore_metrics[col][0],
										   collated_max=zscore_metrics[col][1])
			
	proc_feats.to_parquet(save_path + feat.split('/')[-1].replace('.pickle', '').replace('_raw', '_zscored') + '.parquet')