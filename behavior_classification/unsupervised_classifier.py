#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on 04/07/22

@author: Jorge Iravedra
'''

import numpy as np
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from sklearn.manifold import TSNE
import skimage
import scipy
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import umap


################################################
#			   Class definition				   #
################################################

class BehaviorClustering:

	def __init__(self, path, save_path, feats_list, nframes, attack_frames_path, implanted_list, save_embedding, save_labels):

		self.path = path
		self.save_path = save_path
		self.feats_list = feats_list
		self.nframes = nframes
		self.implanted_list = implanted_list
		self.attack_frames_path = attack_frames_path
		self.save_embedding = save_embedding
		self.save_labels = save_labels

	### Load data ### 

	def open_sessions(self, pickle_path):

		# loads feature pickle

		feats = pd.read_parquet(pickle_path, columns=self.feats_list)

		return feats

	def compile_sessions(self, excepts=False):
		# excepts is a list of mouse ids that you want to embed but not use for training the model

		if excepts: # separate a group of features for training of the model and for the rest of the embedding

			sample_features = {}
			full_features = {}

			feats2sample = [x for x in np.sort(os.listdir(self.path)) if np.array([y in x for y in excepts]).any() == False and '.parquet' in x]
			for session in np.sort(feats2sample):

				if any(imp in session for imp in self.implanted_list):
					n = 4
				else:
					n = 3

				data = self.open_sessions(self.path + session)
				sample_features['_'.join(session.split('_')[:n])] = data
				full_features['_'.join(session.split('_')[:n])] = data

			rest2embed  = [x for x in np.sort(os.listdir(self.path)) if np.array([y in x for y in excepts]).any() == True and '.parquet' in x]

			for session in np.sort(rest2embed):

				if any(imp in session for imp in self.implanted_list):
					n = 4
				else:
					n = 3

				data = self.open_sessions(self.path + session)
				full_features['_'.join(session.split('_')[:n])] = data

			return sample_features, full_features

		else:

			full_features = {}

			feat2sample = [x for x in os.listdir(self.path) if '.parquet' in x]

			for session in np.sort(feats2sample):

				if any(imp in session for imp in self.implanted_list):
					n = 4
				else:
					n = 3

				data = self.open_sessions(self.path + session)
				full_features['_'.join(session.split('_')[:n])] = data

			return full_features

	def save_data(self, data, filename):

		with open(self.save_path + filename, 'wb') as handle:
			pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

	### Clustering ### 

	def uniform_sample(self, feats):

		'''
		
		Uniformly samples features from each video and calculates optimal perplexity for t-SNE

		Inputs:
			feats = feature dictionary containing all sessions
			frames_per_day = int for how many features need to be sampled (e.g. 7142 frames across 56 videos == 400,000 total frames)

		Outputs:
			sampled_feats = array containing collated sample of feature, nrows = frames_per_day * nvideos
			perplexity = int perplexity value for tsne

		'''
		if self.attack_frames_path: 
			n = 20000
		else:
			n = 0
		frames_per_day = (self.nframes - n) / len(list(feats.keys()))
		frames_per_day = int(frames_per_day)
	
		print('Commencing uniform sampling...')
		
		sampled_feats = []
		
		for key in feats.keys():

			sample_idx = np.linspace(0, feats[key].shape[0]-1, frames_per_day).astype(int)
			sampled_feats.append(feats[key].values[sample_idx])
			
		sampled_feats = np.row_stack(sampled_feats) # collate subsampled features

		if self.attack_frames_path:
			attack_frames = pd.read_pickle(self.attack_frames_path)
			attack_frames = attack_frames.sample(n=20000, random_state=42)
			attack_frames = attack_frames[self.feats_list]
			attack_frames = attack_frames.to_numpy() 

			sampled_feats = np.concatenate([sampled_feats, attack_frames], axis=0)
		
		perplexity = np.sqrt(sampled_feats.shape[0]).astype(int)
		
		print('Uniform sampling complete')
		
		return sampled_feats, perplexity

	def tsne_reduce(self, sampled_feats, perp=50, learning_rate=200, exagg=1):

		'''
		
		Extracts t-SNE embedding from sampled data

		Inputs:
			sampled_feats = collated array of feature samples (from uniform or random sampling)
			perp = int for perplexity value
			exag = int for exaggeration value

			when tuning,

		Outputs:
			X_embedded = nsamples * 2 array containing dimensionally reduced feature embedding

		'''
		time_start = time.time()
		print('Commencing TSNE...')
		if exagg:
			X_embedded = TSNE(perplexity=perp, early_exaggeration=exagg, init='pca', learning_rate=learning_rate, random_state=420420).fit_transform(sampled_feats)
		else:
			X_embedded = TSNE(perplexity=perp).fit_transform(sampled_feats)
		print('Features embedded. Time elapsed: %f seconds' % (time.time()-time_start))
		print('Embedded structure shape: ' + str(X_embedded.shape))
		
		return X_embedded

	def umap_reduce(self, sampled_feats, n_neighbors=50, min_dist=0.1, n_components=2): # 0.3, 0.1, 0.01

		'''
		
		Extracts UMAP embedding from sampled data

		Inputs:
			sampled_feats = collated array of feature samples (from uniform or random sampling)
			n_neighbors = int for number of neighboring points used in manifold approximation
			min_dist = float for minimum distance between points in the low-dimensional representation. 
						Higher values make the embedding more focused on preserving the global structure.

		Outputs:
			X_embedded = nsamples * n_components array containing dimensionally reduced feature embedding

		'''
		time_start = time.time()
		print('Commencing UMAP...')
		reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=420420)
		X_embedded = reducer.fit_transform(sampled_feats)
		print('Features embedded. Time elapsed: %f seconds' % (time.time()-time_start))
		print('Embedded structure shape: ' + str(X_embedded.shape))
		
		return X_embedded

	def fit_embedding(self, xtrain, xtest, feats_per_day, save_model=False):
		'''
		Fits embedding to rest of feature set
		Inputs:
		'''

		import datetime

		# Get current date and time
		now = datetime.datetime.now()
		timestamp = now.strftime("%Y%m%d_%H%M%S")

		print('Embeddings being learned...')
		
		# Define the parameter grid
		param_grid = {
			'hidden_layer_sizes': [(500, 250, 125, 50), (1000, 500, 100, 50)],
			'solver': ['adam', 'sgd'],
			'alpha': [0.0001, 0.001],
			'learning_rate': ['constant']
		}
		
		mlp = MLPRegressor()
		grid_search = GridSearchCV(mlp, param_grid, cv=3, verbose=2)
		n = int(xtrain.shape[0]/2)
		
		# fit a perceptron with a training set
		print('Fitting perceptron with training set.')
		grid_search.fit(xtrain[:n], xtest[:n]) # x_train = input, x_embedded = output
		
		# Get the best estimator
		best_mlp = grid_search.best_estimator_
		print('Optimal parameters:', grid_search.best_params_)

		# Save the best model
		if save_model == True:
			joblib.dump(best_mlp, self.save_path + self.save_embedding.split('.')[0] + '_finalModel.pkl')
		
		# make sure it works on a testing set
		o = best_mlp.predict(xtrain[n:])
		plt.figure(figsize=(15,15))
		plt.scatter(o, xtest[n:], s=0.1)
		plt.savefig(self.save_path + self.save_embedding.split('.')[0] + '_MLPfit_' + timestamp + '.png', dpi=200)
		
		# predict embeddings across all days of dataset
		embeddings={}
		
		for test_key in feats_per_day.keys():
			test_X = feats_per_day[test_key]
			print('MLP is predicting %s feature data...' %(test_key))
			embedded_mlp = best_mlp.predict(test_X)
			print('Done')
			embeddings[test_key] = embedded_mlp
			
		print('Embeddings dictionary complete!')
			
		return embeddings

	def heatmap(self, data, axlims=None, bins=100, normed=True, sigma=0.0):
		
		# Initial histogram
		heatmap, xedges, yedges = np.histogram2d(data[:,0], data[:, 1], bins=bins, range=axlims, density=normed)
		
		# Convolve with Gaussian
		heatmap = gaussian_filter(heatmap, sigma=sigma)
		
		return heatmap, xedges, yedges

	def map_density(self, X_embedded, sigma=3.5, percentile=30, sampling_type = 'uniform'):  # changed sigma from 2.5 to 3.5 02/16/24

		# 2.5 and 30 and bins = 50 orig
		# 3.8 and 86.5, bins = 200 2nd iteration; pre 071223
		# 2.5 and 30, bins=200, 071223 led to too many clusters for importance sample. 4 and 30 is moderate, 6 is 20-30? 

		# samplying type should be 'uniform' or 'importance'

		# Find local maxima as "seeds" for the watershed transform
		all_map_density, xe, ye = self.heatmap(X_embedded, bins=200, sigma=sigma)
		density_cutoff = np.percentile(all_map_density, percentile)
		density_mask = all_map_density > density_cutoff
		local_maxes = peak_local_max(all_map_density, indices=False)
		local_maxes[np.logical_not(density_mask)] = False
		markers, n_peaks = scipy.ndimage.label(local_maxes)

		# Compute watershed transform
		labeled_map = watershed(-all_map_density, markers, watershed_line=False)
		labeled_map = labeled_map.astype('float64')
		
		labeled_map_viz = watershed(-all_map_density, markers, watershed_line=True)
		labeled_map_viz = labeled_map_viz.astype('float64')
		
		viz = all_map_density
		viz[labeled_map_viz==0]=0
		plt.figure(figsize=(10,10))
		plt.imshow(all_map_density.T, cmap='rainbow',origin='lower')
		
		plt.axis('off')
		plt.colorbar()
		plt.show()
		if sampling_type == 'uniform':
			plt.savefig(self.save_path + self.save_embedding.split('.')[0] + '_density_uniformSampling.png', dpi=300)
		else: 
			plt.savefig(self.save_path + self.save_embedding.split('.')[0] + '_density_importanceSampling.png', dpi=300)

		plt.figure(figsize=(10,10))
		plt.imshow(labeled_map.T, cmap='rainbow',origin='lower')
		plt.axis('off')

		for i in np.unique(labeled_map.flatten()):
			x=np.mean(np.where(labeled_map==i)[0])
			y=np.mean(np.where(labeled_map==i)[1])
			plt.text(x,y,int(i),size=20)

		plt.show()
		if sampling_type == 'uniform':
			plt.savefig(self.save_path + self.save_embedding.split('.')[0] + '_labeledMap_uniformSampling.png', dpi=300)
		else: 
			plt.savefig(self.save_path + self.save_embedding.split('.')[0] + '_labeledMap_importanceSampling.png', dpi=300)

		return labeled_map, all_map_density, xe, ye

	def embedding_labels(self, embeddings, labeled_map, xe, ye):

		labels = {}

		map_shape=(200,200)
		xmin=min(xe)
		xmax=max(xe)
		ymin=min(ye)
		ymax=max(ye)

		for k, e in embeddings.items():
			
			def to_coords(xy):
				
				x = (xy[0] - xmin) / (xmax - xmin) * map_shape[0]
				y = (xy[1] - ymin) / (ymax - ymin) * map_shape[1]
				if x >= map_shape[0]:
					x = map_shape[0]-1
				if y >= map_shape[1]:
					y = map_shape[1]-1
				if x < 0:
					x=0
				if y < 0:
					y=0
				return int(x), int(y)

			labels[k] = np.array([labeled_map[to_coords(point)] for point in e])
			
		return labels
		
	def embed_sessions(self, X_embedded, embeddings_dictionary, sampling_type):
		
		labeled_map, all_map_density, xe, ye = self.map_density(X_embedded, sampling_type=sampling_type)
		labels = self.embedding_labels(embeddings_dictionary, labeled_map, xe, ye)
		
		return labeled_map, all_map_density, labels

	def random_importance_sample(self, feats): 
		
		'''
		
		importance sampling involves:
		uniform sampling of features -> tsne -> watershed -> then randomly sampling from an n amount of frames 
		corresponding to the clusters identified via watershedding
		
		'''
		frames_per_isample = (self.nframes * 0.8) / (len(list(feats.keys())) * 25)
		frames_per_isample = int(frames_per_isample)
		
		print('>>>> Commencing importance sampling <<<<')
		sfeats, perp = self.uniform_sample(feats) # do uniform sampling
		embs = self.umap_reduce(sfeats)
		# embs = self.tsne_reduce(sfeats, perp=50, learning_rate=400, exagg=1) # play around with tuning here
		embd_dictionary = self.fit_embedding(sfeats, embs, feats) # learn embedding and predict on remaining samples
		_, _, labels = self.embed_sessions(embs, embd_dictionary, sampling_type='uniform') # obtain labels per each cluster using watershed
		
		imp_samples = []
		for key in feats.keys():
		
			clusters = np.unique(labels[key])
			samples_per_day = []

			for cls in clusters:
				idxs = np.where(labels[key]==cls)[0]

				try:
					idxs = np.sort(np.random.choice(idxs, frames_per_isample, replace=False))
					samples_per_day.append(idxs)

				except ValueError:
					idxs = np.sort(np.random.choice(idxs, frames_per_isample, replace=True))
					samples_per_day.append(idxs)
					print('Larger sample than population. Subsampling repeated frames in %s' % key)
					continue

			samples_per_day = np.row_stack(samples_per_day).flatten()
			
			subsampled_frames = feats[key].iloc[samples_per_day].values
			
			imp_samples.append(subsampled_frames)
			
		imp_samples = np.row_stack(imp_samples)
		perplexity = int(np.sqrt(imp_samples.shape[0]))
		
		print('>>>> Importance sampling complete <<<<')
			
		return imp_samples, perplexity

	def full_clustering(self, feats2sample, feats2embed=False):
		
		# importance sampling -> tsne -> learning embedding structure and predictions -> watershed clustering
		
		sfeats, perp = self.random_importance_sample(feats2sample) # do sampling
		embs = self.umap_reduce(sfeats)
		# embs = self.tsne_reduce(sfeats, perp=50, learning_rate=400, exagg=1) # play around with tuning here
		if feats2embed:
			final_feats = feats2embed
		else:
			final_feats = feats2embed
		embd_dictionary = self.fit_embedding(sfeats, embs, final_feats, save_model=True) # learn embedding and predict on remaining samples, save model for future usage
		labeled_map, all_map_density, labels = self.embed_sessions(embs, embd_dictionary, sampling_type='importance') # obtain labels per each cluster
		
		return embd_dictionary, labeled_map, all_map_density, labels

	### Run ### 

	def run(self):

		# write keyword (animal ID or full label) for sessions that you want excluded from the TRAINING set
		excepts_list = ['100566', '100567', '100568', '100569', '103808', '87R2', '86L2', '86L', '87L', '87B', '87L2']
		feats2sample, feats2embed = self.compile_sessions(excepts=excepts_list)
		embs, labeled_map, density, labels = self.full_clustering(feats2sample, feats2embed)

		self.save_data(embs, self.save_embedding)
		self.save_data(labels, self.save_labels)

### Conds ### 

			   # DISTANCES
feats_list1 =  ['proximity', 'resident2intruder head-head', 'resident2intruder head-tti', 'intruder2resident head-tti',
			   # SOCIAL ORIENTATIONS
			   'resident2intruder head2centroid angle', 'intruder2resident head2centroid angle',
			   # SPEEDS
			   'resident centroid roc 100 ms', 'intruder centroid roc 100 ms', 'resident head roc 100 ms', 'intruder head roc 100 ms',
			   # POSTURAL INFORMATION
			   'resident centroid2nose angle', 'intruder centroid2nose angle', 'resident tti2head', 'intruder tti2head']

			   # DISTANCES
feats_list2 =  ['proximity', 'resident2intruder head-head', 'resident2intruder head-tti', 'intruder2resident head-tti',
			   # SOCIAL ORIENTATIONS
			   'resident2intruder head2centroid angle', 'intruder2resident head2centroid angle',
			   # SPEEDS
			   'resident centroid roc 100 ms', 'intruder centroid roc 100 ms', 'resident head roc 100 ms', 'intruder head roc 100 ms',
			   'resident centroid roc 100 ms sum across lags', 'intruder centroid roc 100 ms sum across lags', 'resident head roc 100 ms sum across lags', 'intruder head roc 100 ms sum across lags',
			   # IoU
			   'ious',
			   # POSTURAL INFORMATION
			   'resident centroid2nose angle', 'intruder centroid2nose angle', 'resident tti2head', 'intruder tti2head']

			   # DISTANCES
feats_list3 =  ['proximity', 'resident2intruder head-head', 'resident2intruder head-tti', 'intruder2resident head-tti',
			   # SOCIAL ORIENTATIONS
			   'resident2intruder head2centroid angle', 'intruder2resident head2centroid angle',
			   # SPEEDS
			   'resident centroid roc 100 ms', 'intruder centroid roc 100 ms', 'resident head roc 100 ms', 'intruder head roc 100 ms',
			   'resident centroid roc 100 ms sum across lags', 'intruder centroid roc 100 ms sum across lags', 'resident head roc 100 ms sum across lags', 'intruder head roc 100 ms sum across lags',
			   # IoU and pixel change
			   'ious',
			   # POSTURAL INFORMATION
			   'resident centroid2nose angle', 'intruder centroid2nose angle', 'resident tti2head', 'intruder tti2head']

			   # DISTANCES
feats_list4 =  ['proximity', 'resident2intruder head-head', 'resident2intruder head-tti', 'intruder2resident head-tti',
			   # SOCIAL ORIENTATIONS
			   'resident2intruder head2centroid angle', 'intruder2resident head2centroid angle',
			   # SPEEDS
			   'resident centroid roc 100 ms', 'intruder centroid roc 100 ms', 'resident head roc 100 ms', 'intruder head roc 100 ms', 'proximity roc 100 ms',
			   'resident centroid roc 100 ms sum across lags', 'intruder centroid roc 100 ms sum across lags', 'resident head roc 100 ms sum across lags', 'intruder head roc 100 ms sum across lags',
			   'proximity roc 100 ms sum across lags',
			   # IoU and pixel change
			   'ious',
			   # POSTURAL INFORMATION
			   'resident centroid2nose angle', 'intruder centroid2nose angle', 'resident tti2head', 'intruder tti2head']

			   # DISTANCES
feats_list5 =  ['proximity', 'proximity median across lags', 'resident2intruder head-head', 'resident2intruder head-head median across lags', 'resident2intruder head-tti', 'intruder2resident head-tti',
			   # SOCIAL ORIENTATIONS
			   'resident2intruder head2centroid angle', 'intruder2resident head2centroid angle',
			   # SPEEDS
			   'resident centroid roc 100 ms', 'intruder centroid roc 100 ms', 'resident head roc 100 ms', 'intruder head roc 100 ms', 'proximity roc 100 ms',
			   'resident centroid roc 100 ms sum across lags', 'intruder centroid roc 100 ms sum across lags', 'resident head roc 100 ms sum across lags', 'intruder head roc 100 ms sum across lags',
			   'proximity roc 100 ms sum across lags',
			   # IoU and pixel change
			   'ious',
			   # POSTURAL INFORMATION
			   'resident centroid2nose angle', 'intruder centroid2nose angle', 'resident tti2head', 'intruder tti2head']


### Run ###

np.random.seed(420420)

path = '/scratch/gpfs/jmig/feature_processing/processed_features_020924_parquets/'
save_path = '/scratch/gpfs/jmig/clustering/embeddings_071724/'
frames_path = '/scratch/gpfs/jmig/clustering/new_attack_frames.pickle'

### For parallelizing embedding jobs

feat2map = os.getenv('INPUT_DATA_FILE')
nlists = 5
feat_types = ['Feats%i' % i for i in np.arange(1, nlists+1)] # 
feats_dict = {feat_type: globals().get(f'feats_list{i}') for i, feat_type in enumerate(feat_types, start=1)}
feats_list = feats_dict[feat2map]
# save_embedding = '%s_100k_samples_TSNE_50Perp_400LR_1Exagg_embeddings.pickle' % feat2map # for TSNE
# save_labels = '%s_100k_samples_TSNE_50Perp_400LR_1Exagg_labels.pickle' % feat2map  # for TSNE
save_embedding = '%s_100k_samples_UMAP_01mindist_50neighbors_embeddings.pickle' % feat2map # for UMAP
save_labels = '%s_100k_samples_UMAP_01mindist_50neighbors_labels.pickle' % feat2map  # for UMAP

implanted_list = ['3095', '3096', '3097', '4013', '4014', '4015', '4016', '30R2', '30L', '30B', '29L', '91R2', '87R2', '87L', '86L', '86L2', '87B', '87L2']

BC = BehaviorClustering(path=path, save_path=save_path, feats_list=feats_list, nframes=100000, 
	attack_frames_path=frames_path, implanted_list=implanted_list, save_embedding = save_embedding, save_labels = save_labels)
BC.run()


