'''
Dataloader class with supporting functions for loading configs, optimizing training batch order, etc
'''
import math
import os
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
import gc
import pickle
import re

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow.keras.utils import Sequence

from input.config import NORM_RANGE, SCALE_DICT


def load_config(config_file):
	'''
	Loads configuration with model and dataloader parameters

	Args:
		config_file: string
			YAML filename found in CONFIG_DIR (specified in config.py)
	
	Returns:
		model_config: dict
			dictionary of classification network hyperparameters

		dataloader_config: dict
			dictionary of dataloader parameters
		
		test_dataloader_config: dict
			dictionary of test dataloader parameters
	'''
	with open(os.path.join(config_file)) as f:
		config = yaml.load(f.read(),Loader=yaml.CLoader)
	f.close()

	dataloader_config = config['dataloader']
	# scale_name = dataloader_config.pop('scale')
	# if type(scale_name) is not list:
	# 	dataloader_config.update(scale=SCALE_DICT[scale_name])
	return dataloader_config

def make_FFC(test_dirs,IF_files,OOF_files,ROI=3700,n_samples=100,batches=1,shuffle=True,fname='FFC.npy'):
	'''
	Compute flat-field image based on n_samples images from IF and OOF wells
	
	Args:
		test_dirs: list
			list of directories where IF and OOF files are stored
		
		IF_files: list
			list of in-focus well-level images
		
		OOF_files: list
			list of out-of-focus well-level images

		ROI: int
			dimension of crop inscribed inside the well (well_img.shape == (ROI,ROI))
		
		n_samples: int
			number of samples from IF and OOF to compute FFC from
		
		batches: int
			number of batches to compute FFC in, necessary if n_samples*2 images do not fit in memory
		
		shuffle: bool
			shuffle IF and OOF file lists
		
		fname: string
			desired name of saved FFC image, must have .npy extension

		
	Returns:
		final: np.ndarray
			array with final computed FFC image
	'''
	if shuffle:
		random.shuffle(IF_files)
		random.shuffle(OOF_files)
	final= np.zeros((ROI,ROI,batches))
	for k in range(batches):

		FFC_img = np.zeros((ROI,ROI,n_samples),dtype=np.float32)

		for i in range(k*n_samples,(k+1)*n_samples):
			IF_img = np.array(cv.imread(os.path.join(test_dirs[0],IF_files[i]),cv.IMREAD_UNCHANGED)).astype(np.uint16)

			center = [len(IF_img)//2,len(IF_img[0])//2]
			IF_cropped = IF_img[center[0]-ROI//2:center[0]+ROI//2,center[1]-ROI//2:center[1]+ROI//2]
			
			FFC_img[:,:,i%n_samples] = IF_cropped 
		
		FFC_img = np.min(FFC_img,axis=2)
		final[:,:,k] = FFC_img
	final = np.min(final,axis=2)
	save_npy(final,fname=fname,save_dir='')
	return final

def make_FFC_from_dir(data_dir,channel='ch2',method='mean',ROI=3700,n_samples=200):
	
	
	if channel == 'ch2':
		files = [f for f in os.listdir(data_dir) if re.search('Ch2',f) is not None]
	elif channel == 'ch1':
		files = [f for f in os.listdir(data_dir) if re.search('Ch1',f) is not None]

	if n_samples > len(files):
		n_samples = len(files)
	
	FFC_img = np.zeros((ROI,ROI,n_samples),dtype=np.float32)

	for i in range(n_samples):
		img = np.array(cv.imread(os.path.join(data_dir+files[i]),cv.IMREAD_UNCHANGED)).astype(np.uint16)
		center = [len(img)//2,len(img[0])//2]

		img = img[center[0]-ROI//2:center[0]+ROI//2,center[1]-ROI//2:center[1]+ROI//2]

		FFC_img[:,:,i] = img

	if method == 'mean':
		FFC_img = np.mean(FFC_img,axis=2)
	elif method == 'min':
		FFC_img = np.min(FFC_img, axis=2)

	save_npy(FFC_img,'FFC_from_dir.npy','')
	return FFC_img

def save_npy(img,fname,save_dir):
	'''
	Saves img as .npy file in directory save_dir with name fname
	'''
	with open(save_dir+fname,'wb') as f:
		np.save(f,img)

def center_crop(img,ROI=3700):
	'''
	Crops image with from center with dimension (ROI,ROI)
	'''
	center = [len(img)//2,len(img[0])//2]
	cropped = img[center[0]-ROI//2:center[0]+ROI//2,center[1]-ROI//2:center[1]+ROI//2]

	return cropped

def resize(img,scale):
	'''
		Resize image so that its area is 'scale' the size of the original
	'''
	new_dim = np.sqrt(np.prod(list(img.shape))*scale).astype(int)
	new_shape = (new_dim,new_dim)
	return cv.resize(img,new_shape,interpolation=cv.INTER_AREA)

def get_disjoint_grid_patches(image,patch_size,num_patches,use_center_patches=False,augment=False):
	image = tf.expand_dims(tf.expand_dims(image,-1),0)
	patches = tf.image.extract_patches(images=image,
							sizes=[1, patch_size[0], patch_size[1], 1],
							strides=[1, patch_size[0],patch_size[1], 1],
							rates=[1, 1, 1, 1],
							padding='VALID').numpy().squeeze()
	if use_center_patches:
		if not len(patches.shape) == 1:
			grid_size = np.ceil(np.sqrt(num_patches)).astype(int)
			y_start = np.floor(patches.shape[0]/2).astype(int) - np.ceil(grid_size/2).astype(int)
			x_start = np.floor(patches.shape[1]/2).astype(int) - np.ceil(grid_size/2).astype(int)
			new_patches = np.zeros((num_patches,patches.shape[2]))
			for i in range(num_patches):
				# print(y_start+i//grid_size,x_start+i%grid_size)
				new_patches[i] = patches[y_start+i//grid_size,x_start+i%grid_size]
			patches = new_patches
			patches = patches.reshape((patches.shape[0]*patches.shape[1],patches.shape[2]))
			patches = patches.reshape((patches.shape[0],patch_size[0],patch_size[1]))
		else:
			patches = patches.reshape((patches.shape[0]*patches.shape[1],patches.shape[2]))
			patches = patches.reshape((1,patch_size[0],patch_size[1]))
	else:
		patches = patches.reshape((patches.shape[0]*patches.shape[1],patches.shape[2]))
		if not len(patches.shape) == 1:
			patches = patches.reshape((patches.shape[0],patch_size[0],patch_size[1]))
		else:
			patches = patches.reshape((1,patch_size[0],patch_size[1]))
	if augment:
		rot_k= random.choices(np.arange(1,3),k=num_patches)
		flip= random.choices(np.arange(0,2),k=num_patches)
		for i in range(num_patches):
			patch = patches[i]
			patch = np.rot90(patch,k=rot_k[i])
			if flip[i] ==1:
				patch = np.fliplr(patch)
			
			patches[i] = patch
	patches_left = num_patches-len(patches)
	return patches, patches_left

def get_random_patches(image, patch_size, num_patches,augment=True):
	'''
	Extracts patches randomly from an image with an option to augment them
	via random rotations and reflections
	'''
	dim = image.shape[0]
	patches = np.zeros((num_patches,patch_size[0],patch_size[1]))

	for i in range(num_patches):
		top_left = (random.randint(0,dim-patch_size[0]),random.randint(0,dim-patch_size[1]))
		patch = image[top_left[0]:top_left[0]+patch_size[0],top_left[1]:top_left[1]+patch_size[1]]
		patches[i] = patch
	if augment:
		rot_k= random.choices(np.arange(1,3),k=num_patches)
		flip= random.choices(np.arange(0,2),k=num_patches)
		for i in range(num_patches):
			patch = patches[i]
			patch = np.rot90(patch,k=rot_k[i])
			if flip[i] ==1:
				patch = np.fliplr(patch)
			patches[i] = patch
	return patches

def make_patches(image,patch_size,num_patches, use_global_normalization=False,randomize_location=False,use_center_patches=False,augment=False,seed=None,patch_selection_method='mid_std_dev'):
	'''
	Create patches from an image

	Args:
		image: np.ndarray
			2D numpy array of a cropped well-level image

		patch_size: tuple
			desired size of patches in pixels (e.g., (100,100))
		
		num_patches: int
			number of patches to return

		use_global_normalization: bool
			if True, normalize patches using NORM_RANGE defined in config.py
			if False, normalize patches at well-level

		randomize_location: bool
			if True, randomize location of patches
			if False, patches will be taken from a disjoint grid of tiles of size patch_size and supplements 
				from random locations when the grid doesn't contain enough patches

		use_center_patches: bool
			if True, take images from the center rather than sorting by standard deviation

		augment: bool
			if True, randomly rotate and reflect patches

		seed: int
			seed for pseudorandom number generation used
			used in data augmentation

	Returns:
		final_patches: np.ndarray
			3D stack of sorted patches (sample shape: (9,100,100))
	'''

	if randomize_location:
		patches = get_random_patches(image,patch_size,num_patches,augment=augment)
	else:

		patches, patches_left = get_disjoint_grid_patches(image,patch_size,num_patches,use_center_patches=use_center_patches,augment=augment)
		if patches_left > 0:
			random_patches = get_random_patches(image,patch_size,num_patches=patches_left,augment=True)
			patches= np.concatenate((patches,random_patches),axis=0)


	# Set normalization range
	if use_global_normalization:
		norm_range = NORM_RANGE
		patches[patches<norm_range[0]] = norm_range[0]
	else:
		norm_range = (np.min(patches),np.max(patches))

	# Normalize patches and sort by standard deviation
	patches = (patches-norm_range[0])/(norm_range[1]-norm_range[0])
	std_devs = np.std(patches,axis=(1,2))
	sort_ids = np.argsort(std_devs)[::-1]
	patches = (patches[sort_ids])
	
	# Throw away unwanted patches
	if num_patches < len(patches):
		if patch_selection_method == 'max_std_dev':
			final_patches = patches[:num_patches]
		elif patch_selection_method == 'mid_std_dev':
			mid_id = np.round(len(patches)/2).astype(int)
			half = int(num_patches/2)
			final_patches = patches[mid_id-half:mid_id+half]
		elif patch_selection_method == 'random':
			rand_ids = np.random.choice(np.arange(len(patches)),size=num_patches,replace=False)
			final_patches = patches[rand_ids]
	elif num_patches == len(patches):
		final_patches = patches
	else:
		num_patches = len(patches)
		final_patches = patches

	return final_patches, len(final_patches)


def get_center_patch(image,patch_size):
	'''
	Generates single patch from center of well-level image
	'''
	coords = (image.shape[0]//2,image.shape[1]//2)
	center_patch = image[coords[0]-patch_size[0]//2:coords[0]+patch_size[0]//2,coords[1]-patch_size[1]//2:coords[1]+patch_size[1]//2]

	center_patch = (center_patch-np.min(center_patch))/(np.max(center_patch)-np.min(center_patch))
	center_patch = center_patch.reshape((1,patch_size[0],patch_size[1]))
	return center_patch

class DeploymentDataGenerator(Sequence):
	'''
	Data Generator for Focus Analysis Deep Learning Network implemented in foca.py

	Methods:

		__init__: None
			initialize data generator

		__len__: int
			returns number of batches

		__getitem__:
			returns batch idx


		
	'''
	current_batch = None

	def __init__(self,df,batch_size,patch_size=(150,150,1),
					patches_per_well=5,ROI=648,FFC=None,use_center_patches=False,patch_selection_method='mid_std_dev'):

		self.batch_size = batch_size
		self.patch_size = patch_size
		self.patches_per_well = patches_per_well
		self.df = df
		self.use_center_patches = use_center_patches
		self.patch_selection_method = patch_selection_method
		self.FFC = FFC
		if self.FFC is not None:
			self.FFC_img = np.load(self.FFC,allow_pickle=True)
		else:
			self.FFC_img = None
		self.ROI = ROI
		self.current_batch = None

	def __len__(self):
		return math.ceil(len(self.df)/float(self.batch_size))
	
	def __getitem__(self,idx):
		batch_df = self.df.iloc[idx*self.batch_size:(idx+1)*self.batch_size]
		self.current_batch = batch_df

		for i in range(len(batch_df)):
			well_info = dict(batch_df.iloc[i])

			# print(well_info['Filename'])
			img = cv.imread(os.path.join(well_info['Location'],well_info['Filename']),
							cv.IMREAD_GRAYSCALE).astype(np.float32)
			# FOR 6um configuration
			if img.shape[0]> 1300:
				img = cv.resize(img,(1296,1296),interpolation=cv.INTER_AREA)

			if self.ROI is not None:
				img = center_crop(img,ROI=self.ROI)
				if self.FFC_img is not None:
					FFC_crop = center_crop(self.FFC_img,ROI=self.ROI)

			if self.FFC is not None:
				prev_range = (np.min(img),np.max(img))
				img = img/(FFC_crop+np.finfo(float).eps)
				img = (img-np.min(img))/(np.max(img)-np.min(img))*(prev_range[1]-prev_range[0])+prev_range[0]
			# img = ((img-np.min(img))/(np.max(img)-np.min(img)))
	
			patches,_ = make_patches(img,self.patch_size,self.patches_per_well,randomize_location=False,
									use_center_patches=self.use_center_patches,
									patch_selection_method=self.patch_selection_method)
			
			if i == 0:
				batch_patches = patches
			else:
				batch_patches = np.concatenate((batch_patches,patches))

		batch_patches = tf.expand_dims(batch_patches,axis=-1)
		return batch_patches
