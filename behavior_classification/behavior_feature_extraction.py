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
from scipy.signal import savgol_filter

def fill_missing(Y, kind='cubic'):
	
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

def filter_XY(df):
	filtered_df = df.copy()
	for column in df.columns:
		filtered_df[column] = savgol_filter(df[column], window_length=21, polyorder=3)
	return filtered_df

def calculate_iou(box_1, box_2, mode = 'mouse'):
	
	'''
	IoU = intersection over union. this is a metric for quantifying bounding box overlap. here
	we're applying it to quantify the overlap between the mice bounding boxes
	
	though we're not specifically using IoU because values in IoU are not comparable with each other if the 
	bounding boxes are different per condition. so here we're doing the intersection of the bounding boxes
	over the area of the mouse bounding box
	
	this function uses the shapely toolkit to create a Polygon object, over which we can quantify the
	intersection in the area of both bounding boxes, and normalize that by the area of the mouse bounding box.
	then we set a threshold of iou = 0.5.
	
	input: 
		box 1 = m1 box [xmin, ymin, xmax, ymax]
		box 2 = m2 box [xmin, ymin, xmax, ymax]
	output:
		iou = float describing box 1 and 2 overlap
	
	'''
	
	from shapely import geometry as g
	poly_1 = g.box(*box_1, ccw=True)
	poly_2 = g.box(*box_2, ccw=True)
	if mode == 'mouse':
		iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
	else: 
		iou = poly_1.intersection(poly_2).area / poly_2.area
		
	return iou

def bbox_features(slpdata, resh=11):
	
	# uses the BoundingBox function from planar to fit bounding box per frame for each mouse
	
	top_m1_nodes = []
	top_m2_nodes = [] 

	for col in slpdata.columns.levels[0]:

		if 'top_m1' in col and 't0' not in col and 't1' not in col and 't2' not in col and 'tailtip' not in col:
			top_m1_nodes.append(col)
		elif 'top_m2' in col and 't0' not in col and 't1' not in col and 't2' not in col and 'tailtip' not in col:
			top_m2_nodes.append(col)
	
	# initialize features we want to capture using bounding boxes
	
	m1_centroid = []
	m2_centroid = []
	ious = []
	
	for i in range(slpdata.shape[0]):
		
		# define bounding box for m1
		m1_bbox = BoundingBox(slpdata[np.asarray(top_m1_nodes)].to_numpy()[i].reshape(resh, 2))
		
		# extract bounding box points for overlap calculation
		m1_arr = np.zeros(4)
		m1_arr[0] = m1_bbox.min_point[0]
		m1_arr[1] = m1_bbox.min_point[1]
		m1_arr[2] = m1_bbox.max_point[0]
		m1_arr[3] = m1_bbox.max_point[1]
		
		# define bounding box for m2
		m2_bbox = BoundingBox(slpdata[np.asarray(top_m2_nodes)].to_numpy()[i].reshape(resh, 2))
		
		# extract bounding box points for overlap calculation
		m2_arr = np.zeros(4)
		m2_arr[0] = m2_bbox.min_point[0]
		m2_arr[1] = m2_bbox.min_point[1]
		m2_arr[2] = m2_bbox.max_point[0]
		m2_arr[3] = m2_bbox.max_point[1]
		
		# extract features
		m1_centroid.append(m1_bbox.center)
		m2_centroid.append(m2_bbox.center)
		ious.append(calculate_iou(m1_arr, m2_arr))
			
	m1_centroid = np.asarray(m1_centroid)
	m2_centroid = np.asarray(m2_centroid)
	ious = np.asarray(ious)

	return m1_centroid, m2_centroid, ious

def calculate_orientation_angle(a, b, c):
	
	# a = neck or trunk
	# b = head or nose
	# c = intruder target (intruder head or centroid)

	crossprod = (c['x']-b['x'])*(b['y']-a['y']) - (c['y']-b['y'])*(b['x']-a['x'])
	dotprod = (c['x']-b['x'])*(b['x']-a['x']) + (c['y']-b['y'])*(b['y']-a['y'])
	angles = np.arctan2(crossprod, dotprod)
	angles = np.where(angles < 0, angles * -1, angles)
	angles = np.array([np.degrees(angle) for angle in angles])
	
	return angles

def slp_features(pickle, implanted_list, savepath, videopath=False):
	
	'''
	outputs features dictionary to run tsne on
	
	INPUTS: 
		slpdata = Df output from slp_load
		video_path = str path to video you wish you extract coordinates from
	OUTPUTS:
		features = 
	'''

	print('Initiating feature extraction of %s' % pickle)

	slpdata = pd.read_pickle(pickle)
	slpdata = filter_XY(slpdata)
	slpdata = slpdata.rename(columns={'top_m1_tail_tip':'top_m1_tailtip', 'top_m2_tail_tip':'top_m2_tailtip', 'top_m1_tti':'top_m1_tailbase', 'top_m2_tti':'top_m2_tailbase'})
	
	if any(animal in pickle for animal in implanted_list):
		n = 4
	else:
		n = 3
	
	features = {}
	nodes = list(slpdata.columns.levels[0])
	
	print('### Obtaining bounding box and centroid data ###')
	m1_centroid, m2_centroid, ious = bbox_features(slpdata)

	### Clean up data for last time ### 

	new_cols = pd.MultiIndex.from_product([['top_m1_centroid', 'top_m2_centroid'],['x','y']])
	nans = np.zeros((slpdata.shape[0], 4))
	nans[:] = np.nan

	df = pd.DataFrame(nans, columns=new_cols)
	slpdata = pd.concat([slpdata, df], axis=1)
	slpdata['top_m1_centroid'] = m1_centroid
	slpdata['top_m2_centroid'] = m2_centroid

	print('### Done ###')


	####################    
	### feature list ###### TOP VARIABLES
	####################

	print("### Enumerating top variables ###")

	mouse_ids = ['top_m1_', 'top_m2_'] # allows iterating through both mice to have repeated features
	feat_labels =['resident ', 'intruder ']

	for m, mouse in zip(mouse_ids, feat_labels): # iterate the below features per mouse
			
	### Relevant inter-node distances ### 

		features[mouse+'tti2head'] = np.linalg.norm(slpdata[m+'tailbase']-slpdata[m+'head'],axis=1)
		features[mouse+'tti2trunk'] = np.linalg.norm(slpdata[m+'tailbase']- slpdata[m+'trunk'],axis=1)
		features[mouse+'tti2neck'] = np.linalg.norm(slpdata[m+'tailbase']- slpdata[m+'neck'],axis=1)
		features[mouse+'tti2centroid'] = np.linalg.norm(slpdata[m+'tailbase']- slpdata[m+'centroid'],axis=1)
		features[mouse+'head2nose'] = np.linalg.norm(slpdata[m+'head']- slpdata[m+'nose'],axis=1)
		features[mouse+'trunk2head'] = np.linalg.norm(slpdata[m+'trunk']- slpdata[m+'head'],axis=1)
		features[mouse+'forepaw left2head'] = np.linalg.norm(slpdata[m+'forepaw_left']- slpdata[m+'head'],axis=1)
		features[mouse+'forepaw right2head'] = np.linalg.norm(slpdata[m+'forepaw_right']- slpdata[m+'head'],axis=1)
		features[mouse+'forepaw left2trunk'] = np.linalg.norm(slpdata[m+'forepaw_left']-slpdata[m+'trunk'],axis=1)
		features[mouse+'forepaw right2trunk'] = np.linalg.norm(slpdata[m+'forepaw_right']-slpdata[m+'trunk'],axis=1)
		features[mouse+'hindpaw left2trunk'] = np.linalg.norm(slpdata[m+'hindpaw_left']- slpdata[m+'trunk'],axis=1)
		features[mouse+'hindpaw right2trunk'] = np.linalg.norm(slpdata[m+'hindpaw_right']-slpdata[m+'trunk'],axis=1)
		features[mouse+'hindpaw left2tti'] = np.linalg.norm(slpdata[m+'hindpaw_left']- slpdata[m+'tailbase'],axis=1)
		features[mouse+'hindpaw right2tti'] = np.linalg.norm(slpdata[m+'hindpaw_right']-slpdata[m+'tailbase'],axis=1)
		features[mouse+'hindpaw left2right'] = np.linalg.norm(slpdata[m+'hindpaw_left']-slpdata[m+'hindpaw_right'],axis=1)
		features[mouse+'forepaw left2right'] = np.linalg.norm(slpdata[m+'forepaw_left']-slpdata[m+'forepaw_right'],axis=1)
		
	 ### Relevant within-mouse angles ### 
	
		features[mouse+'tailbase2head angle'] =calculate_orientation_angle(slpdata[m+'tailbase'], # tailbase 2 neck 2 head (body direction)
																			slpdata[m+'neck'],
																			slpdata[m+'head'])
		
		features[mouse+'tailbase2nose angle'] =calculate_orientation_angle(slpdata[m+'tailbase'], # tailbase 2 neck 2 nose (body direction)
																			slpdata[m+'neck'],
																			slpdata[m+'nose'])
		
		features[mouse+'centroid2nose angle'] =calculate_orientation_angle(slpdata[m+'centroid'], # centroid 2 head 2 nose (head direction)
																			slpdata[m+'head'],
																			slpdata[m+'nose'])
		
		features[mouse+'neck2nose angle'] =calculate_orientation_angle(slpdata[m+'neck'], # neck 2 head 2 nose (head direction)
																			slpdata[m+'head'],
																			slpdata[m+'nose'])
		
		features[mouse+'tailbase2trunk angle'] =calculate_orientation_angle(slpdata[m+'tailbase'], # tailbase 2 trunk 2 head (body direction)
																			slpdata[m+'trunk'],
																			slpdata[m+'head'])
		
	### ROC for above variables ### 

		print('### Estimating ROC values ###')
	
		for stp in [2, 4, 20, 40, 80, 160, 200]:
		
		### Inter-node distance ROCs ### 

			features[mouse+'tti2head roc %i ms' % (stp / 40 * 1000)] = np.concatenate([np.zeros(stp), features[mouse+'tti2head'][stp:] - features[mouse+'tti2head'][:-stp]])
			features[mouse+'tti2head abs roc %i ms' % (stp / 40 * 1000)] = np.abs(np.concatenate([np.zeros(stp), features[mouse+'tti2head'][stp:] - features[mouse+'tti2head'][:-stp]]))

			features[mouse+'tti2neck roc %i ms' % (stp / 40 * 1000)] = np.concatenate([np.zeros(stp), features[mouse+'tti2neck'][stp:] - features[mouse+'tti2neck'][:-stp]])
			features[mouse+'tti2neck abs roc %i ms' % (stp / 40 * 1000)] = np.abs(np.concatenate([np.zeros(stp), features[mouse+'tti2neck'][stp:] - features[mouse+'tti2neck'][:-stp]]))


		### Relevant locomotive features ### 

			features[mouse+'centroid roc %i ms' % (stp / 40 * 1000)] = np.concatenate([np.zeros(stp), np.linalg.norm(slpdata[m+'centroid'].values.T[:, stp:] - slpdata[m+'centroid'].values.T[:,:-stp], axis=0)])
			features[mouse+'head roc %i ms' % (stp / 40 * 1000)] = np.concatenate([np.zeros(stp),  np.linalg.norm(slpdata[m+'head'].values.T[:, stp:] - slpdata[m+'head'].values.T[:,:-stp], axis=0)])
			features[mouse+'tti roc %i ms' % (stp / 40 * 1000)] = np.concatenate([np.zeros(stp), np.linalg.norm(slpdata[m+'tailbase'].values.T[:, stp:] - slpdata[m+'tailbase'].values.T[:,:-stp], axis=0)])
			features[mouse+'t0 roc %i ms' % (stp / 40 * 1000)] = np.concatenate([np.zeros(stp), np.linalg.norm(slpdata[m+'t0'].values.T[:, stp:] - slpdata[m+'t0'].values.T[:,:-stp], axis=0)])
			features[mouse+'t1 roc %i ms' % (stp / 40 * 1000)] = np.concatenate([np.zeros(stp), np.linalg.norm(slpdata[m+'t1'].values.T[:, stp:] - slpdata[m+'t1'].values.T[:,:-stp], axis=0)])
			features[mouse+'t2 roc %i ms' % (stp / 40 * 1000)] = np.concatenate([np.zeros(stp), np.linalg.norm(slpdata[m+'t2'].values.T[:, stp:] - slpdata[m+'t2'].values.T[:,:-stp], axis=0)])
			features[mouse+'tailtip roc %i ms' % (stp / 40 * 1000)] = np.concatenate([np.zeros(stp), np.linalg.norm(slpdata[m+'tailtip'].values.T[:, stp:] - slpdata[m+'tailtip'].values.T[:,:-stp], axis=0)])
			features[mouse+'avg tail roc %i ms' % (stp / 40 * 1000)] = np.mean([features[mouse+'tailtip roc %i ms' % (stp / 40 * 1000)], features[mouse+'t2 roc %i ms' % (stp / 40 * 1000)], features[mouse+'t1 roc %i ms' % (stp / 40 * 1000)], features[mouse+'t0 roc %i ms' % (stp / 40 * 1000)]], axis=0)
			
		### Locomotive feature differences

			features[mouse+'tail vs centroid roc %i ms' % (stp / 40 * 1000)] = features[mouse+'avg tail roc %i ms' % (stp / 40 * 1000)] - features[mouse+'centroid roc %i ms' % (stp / 40 * 1000)]
			features[mouse+'head vs tti roc %i ms' % (stp / 40 * 1000)] = features[mouse+'head roc %i ms' % (stp / 40 * 1000)] - features[mouse+'tti roc %i ms' % (stp / 40 * 1000)]

		### Within-mouse angle change ### 

			features[mouse+'tailbase2head angle roc %i ms' % (stp / 40 * 1000)] = np.concatenate([np.zeros(stp), features[mouse+'tailbase2head angle'][stp:] - features[mouse+'tailbase2head angle'][:-stp]])
			features[mouse+'tailbase2head angle abs roc %i ms' % (stp / 40 * 1000)] = np.abs(np.concatenate([np.zeros(stp), features[mouse+'tailbase2head angle'][stp:] - features[mouse+'tailbase2head angle'][:-stp]]))

			features[mouse+'tailbase2nose angle roc %i ms' % (stp / 40 * 1000)] = np.concatenate([np.zeros(stp), features[mouse+'tailbase2nose angle'][stp:] - features[mouse+'tailbase2nose angle'][:-stp]])
			features[mouse+'tailbase2nose angle abs roc %i ms' % (stp / 40 * 1000)] = np.abs(np.concatenate([np.zeros(stp), features[mouse+'tailbase2nose angle'][stp:] - features[mouse+'tailbase2nose angle'][:-stp]]))

			features[mouse+'centroid2nose angle roc %i ms' % (stp / 40 * 1000)] = np.concatenate([np.zeros(stp), features[mouse+'centroid2nose angle'][stp:] - features[mouse+'centroid2nose angle'][:-stp]])
			features[mouse+'centroid2nose angle abs roc %i ms' % (stp / 40 * 1000)] = np.abs(np.concatenate([np.zeros(stp), features[mouse+'centroid2nose angle'][stp:] - features[mouse+'centroid2nose angle'][:-stp]]))

			features[mouse+'neck2nose angle roc %i ms' % (stp / 40 * 1000)] = np.concatenate([np.zeros(stp), features[mouse+'neck2nose angle'][stp:] - features[mouse+'neck2nose angle'][:-stp]])
			features[mouse+'neck2nose angle abs roc %i ms' % (stp / 40 * 1000)] = np.abs(np.concatenate([np.zeros(stp), features[mouse+'neck2nose angle'][stp:] - features[mouse+'neck2nose angle'][:-stp]]))

			features[mouse+'tailbase2trunk angle roc %i ms' % (stp / 40 * 1000)] = np.concatenate([np.zeros(stp), features[mouse+'tailbase2trunk angle'][stp:] - features[mouse+'tailbase2trunk angle'][:-stp]])
			features[mouse+'tailbase2trunk angle abs roc %i ms' % (stp / 40 * 1000)] = np.abs(np.concatenate([np.zeros(stp), features[mouse+'tailbase2trunk angle'][stp:] - features[mouse+'tailbase2trunk angle'][:-stp]]))
	
		print('### ROC values finalized ###')

	### Social features ### 
	
	# IoU
	
	features['ious'] = ious
	
	# 0 to 180 deg orientation
	
	features['resident2intruder head2head angle'] = calculate_orientation_angle(slpdata['top_m1_neck'],
																				slpdata['top_m1_head'],
																				slpdata['top_m2_head'])
	
	features['resident2intruder nose2head angle'] = calculate_orientation_angle(slpdata['top_m1_neck'], 
																				slpdata['top_m1_nose'], 
																				slpdata['top_m2_head'])
	
	features['resident2intruder head2tti angle'] = calculate_orientation_angle(slpdata['top_m1_neck'],
																			   slpdata['top_m1_head'],
																			   slpdata['top_m2_tailbase'])
	
	features['resident2intruder nose2tti angle'] = calculate_orientation_angle(slpdata['top_m1_neck'], 
																			   slpdata['top_m1_nose'], 
																			   slpdata['top_m2_tailbase'])
	
	features['resident2intruder head2centroid angle'] = calculate_orientation_angle(slpdata['top_m1_neck'],
																				slpdata['top_m1_head'],
																				slpdata['top_m2_centroid'])
	
	features['resident2intruder nose2centroid angle'] = calculate_orientation_angle(slpdata['top_m1_neck'], 
																				slpdata['top_m1_nose'], 
																				slpdata['top_m2_centroid'])
	
	features['intruder2resident head2head angle'] = calculate_orientation_angle(slpdata['top_m2_neck'],
																				slpdata['top_m2_head'],
																				slpdata['top_m1_head'])
	
	features['intruder2resident nose2head angle'] = calculate_orientation_angle(slpdata['top_m2_neck'], 
																				slpdata['top_m2_nose'], 
																				slpdata['top_m1_head'])
	
	features['intruder2resident head2tti angle'] = calculate_orientation_angle(slpdata['top_m2_neck'],
																			   slpdata['top_m2_head'],
																			   slpdata['top_m1_tailbase'])
	
	features['intruder2resident nose2tti angle'] = calculate_orientation_angle(slpdata['top_m2_neck'], 
																			   slpdata['top_m2_nose'], 
																			   slpdata['top_m1_tailbase'])
	
	features['intruder2resident head2centroid angle'] = calculate_orientation_angle(slpdata['top_m2_neck'],
																			   slpdata['top_m2_head'],
																			   slpdata['top_m1_centroid'])
	
	features['intruder2resident nose2centroid angle'] = calculate_orientation_angle(slpdata['top_m2_neck'], 
																			   slpdata['top_m2_nose'], 
																			   slpdata['top_m1_centroid'])
	
	# proximity and locomotive social features
																				  
	intercentroid_distance = np.linalg.norm(slpdata['top_m1_centroid'] - slpdata['top_m2_centroid'], axis=1)
	features['proximity'] = intercentroid_distance
	
	for stp in [2, 4, 20, 40, 80, 160, 200]:

		# derivative of distance
		features['proximity roc %i ms' % (stp / 40 * 1000)] = np.concatenate([np.zeros(stp), intercentroid_distance[stp:] - intercentroid_distance[:-stp]])
		features['proximity abs roc %i ms' % (stp / 40 * 1000)] = np.abs(np.concatenate([np.zeros(stp), intercentroid_distance[stp:] - intercentroid_distance[:-stp]]))

		# derivative of ious
		features['ious roc %i ms' % (stp / 40 * 1000)] = np.concatenate([np.zeros(stp), ious[stp:] - ious[:-stp]])
		features['ious abs roc %i ms' % (stp / 40 * 1000)] = np.abs(np.concatenate([np.zeros(stp), ious[stp:] - ious[:-stp]]))

		# difference between mice speeds
		features['centroid roc differential %i ms' % (stp / 40 * 1000)] = features['resident centroid roc %i ms' % (stp / 40 * 1000)] - features['intruder centroid roc %i ms' % (stp / 40 * 1000)]
		features['head roc differential %i ms' % (stp / 40 * 1000)] = features['resident head roc %i ms' % (stp / 40 * 1000)] - features['intruder head roc %i ms' % (stp / 40 * 1000)]

	# head-based distances

	features['resident2intruder head-head'] = np.linalg.norm(slpdata['top_m1_head'] - slpdata['top_m2_head'],axis=1)

	features['resident2intruder head-tti'] = np.linalg.norm(slpdata['top_m1_head']-slpdata['top_m2_tailbase'],axis=1)

	features['resident2intruder head-centroid'] = np.linalg.norm(slpdata['top_m1_head']-slpdata['top_m2_centroid'],axis=1)

	features['intruder2resident head-centroid'] = np.linalg.norm(slpdata['top_m2_head']- slpdata['top_m1_centroid'],axis=1)

	features['intruder2resident head-tti'] = np.linalg.norm(slpdata['top_m2_head']-slpdata['top_m1_tailbase'],axis=1)

	# nose-based distances
	
	features['resident2intruder nose-nose'] = np.linalg.norm(slpdata['top_m1_nose'] - slpdata['top_m2_nose'],axis=1)

	features['resident2intruder nose-tti'] = np.linalg.norm(slpdata['top_m1_nose']-slpdata['top_m2_tailbase'],axis=1)

	features['resident2intruder nose-centroid'] = np.linalg.norm(slpdata['top_m1_nose']-slpdata['top_m2_centroid'],axis=1)

	features['intruder2resident nose-centroid'] = np.linalg.norm(slpdata['top_m2_nose']- slpdata['top_m1_centroid'],axis=1)

	features['intruder2resident nose-tti'] = np.linalg.norm(slpdata['top_m2_nose']-slpdata['top_m1_tailbase'],axis=1)

	# lag features
	time_lags = np.linspace(-40*2, 40*2, 14).astype(int)
	for col in list(features.keys()):
		feat = features[col]
		lagged_data = pd.DataFrame()
		for lag in time_lags:
			# Shift the time series data by the lag value
			lagged_data['%i ms lag' % int(lag*1000/40)] = pd.Series(feat).shift(lag)
		lagged_data.fillna(0, inplace=True)
		features[' '.join([col, 'mean across lags'])] = lagged_data.mean(axis=1)
		features[' '.join([col, 'median across lags'])] = lagged_data.median(axis=1)
		features[' '.join([col, 'sum across lags'])] = lagged_data.sum(axis=1)
	
	print('### Done ###')
	print('### Processing features ###')

	# data formatting from dictionary
	feature_ids = list(features.keys())

	# features to be processed
	features_array = list(features.values())
	features_array = pd.DataFrame(features_array).T 
	features_array.columns = list(features.keys())

	print('### Done ###')
	print('### Saving feature file ###')
	print('_'.join(pickle.split('/')[-1].split('_')[:n]))
	features_array.to_pickle(savepath + '_'.join(pickle.split('/')[-1].split('_')[:n]) + '_raw_features.pickle')

### FOR TESTING

pickle_path = '/scratch/gpfs/jmig/feature_processing/pickles/'
# pickle_files = [pickle_path + x for x in os.listdir(pickle_path)]
# pickle_files = [pickle_path + x for x in os.listdir(pickle_path) if any(s in x for s in ['927R', '927L', '933R', '583L2', '583B'])]
implanted_list = ['3095', '3096', '3097', '4013', '4014', '4015', '4016', '91R2', '30L', '30B', '29L', '30R2', '87L', '86L2', '87R2', '86L', '87B', '87L2', '927R', '927L', '933R', '583L2', '583B']
savepath = '/scratch/gpfs/jmig/feature_processing/extracted_features/'
threads = 16

### SUBMIT

pickle = os.getenv('INPUT_DATA_FILE')
slp_features(pickle, implanted_list=implanted_list, savepath=savepath)