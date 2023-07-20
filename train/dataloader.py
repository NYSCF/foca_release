'''
Datagenerator classes for training,validation, and testing focal analysis DNN
along with supporting functions for preprocessing images
and organizing training batches
'''
import os

import numpy as np
import pandas as pd
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import random
import re
import sys
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Sequence

sys.path.append(str(Path(__file__).parents[1]))
from input.config import DATASET_CSV_DIR, FFC_DIR, NORM_RANGE
from models.foca import FocA


def make_FFC(data_dir,blur=True):
	print(FFC_DIR)
	files = [f for f in os.listdir(data_dir) if re.search('Ch2',f) is not None and f.endswith('.tiff')]
	imgs = np.zeros((96,1296,1296))
	for i in range(len(files)):
		# print(i)
		img = cv.imread(os.path.join(data_dir,files[i]),cv.IMREAD_GRAYSCALE)
		img = cv.resize(img,(1296,1296)).astype(np.float32)
		imgs[i] = img
		# if i==0:
		# 	FFC = img
		# else:
		# 	FFC= FFC.astype(np.float32)
		# 	FFC = (FFC*i+img)/(i+1)
		# print(np.max(FFC))
	FFC = np.mean(imgs,axis=0)
	if blur:
		# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11))
		FFC = cv.GaussianBlur(FFC,(21,21),0)
	np.save(os.path.join(FFC_DIR,'empty_blur.npy'),FFC,allow_pickle=True)
def get_patches_per_class(min_patches,class_dist):
	'''
	Calculate number of patches to take per class to even out class imbalance in a training batch

	Args:
		min_patches: int
			minimum number of patches to take per image, the class with the most samples will have this many patches

		class_dist: list
			list containing number of samples in each class

	Return:
		patches_per_class: list
			list containing number to take for each class in a training batch in order to have the same number of patches
			from each class
	'''
	patches_per_class = []
	max_class = np.max(class_dist)
	for c in range(len(class_dist)):
		patches_per_class.append(int(min_patches*(max_class/(class_dist[c]+np.finfo(float).eps))))
	return patches_per_class



def resize(img,scale=None,new_size=None):
	'''
	Resize img based on scale or prescribed new size

	Args:
		img: np.ndarray

		scale: float
			Optional: if specified, resize the image by scaling width and height by this
			Ex: (100,100) image, scale=0.5 returns (50,50) image
			Overrides new_size argument

		new_size: tuple
			Optional: if specified, resize the image to this shape
			Ex: (100,100) image, new_size= (25,25) return (25,25) image

	Return:
		resized_img: np.ndarray
			img resized to desired shape
	'''
	new_shape = img.shape
	if scale is not None:
		new_dim = int(scale*img.shape[0])
		new_shape = (new_dim,new_dim)
	elif new_size is not None:
		new_shape = new_size
	else:
		return img

	resized_img = cv.resize(img,new_shape,interpolation=cv.INTER_AREA)
	return resized_img

def center_crop(img,ROI):
	'''
	Crop an image around its centered with specified ROI

	Args:
		img: np.ndarray
			2D numpy array of full well-level image

		ROI: int
			dimension of desired crop

	Return:
		cropped: np.ndarray
			2D numpy array of cropped well-level image with size (ROI,ROI)
	'''
	center = [len(img)//2,len(img[0])//2]
	cropped = img[center[0]-ROI//2:center[0]+ROI//2,center[1]-ROI//2:center[1]+ROI//2]

	return cropped

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
		if patches_left >0:
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
	# for i in range(len(final_patches)):
	# 	patch = final_patches[i]
	# 	# randomize contrast
	# 	boundary = random.uniform(0.2,0.6)
	# 	lower, upper = random.uniform(0,boundary),random.uniform(boundary,1)

	# 	patch = (patch-np.min(patch))/(np.max(patch)-np.min(patch))*(upper-lower)+lower
	# 	final_patches[i] = patch
	return final_patches


def make_patch_df(well_df, wells_per_stage, min_patches, batch_size):
	'''
	Construct hierarchical DataFrame with sorted patches:
	Hierarchy:
		1. Stage
		2. Class
		3. Plate
		4. Filename
		5. Patch
	We call groups of wells preprocessed into patches and loaded into memory "stages"

	Class is a binary label 0 or 1 where 0 is in-focus and 1 is out-of-focus
	Plate is a plate barcode we use to identify individual plates
	Filename 

	Args:
		well_df: pd.DataFrame
			DataFrame of well images with fields: directory, filename, plate, class 

		wells_per_stage: int
			Number of well images loaded and preprocessed in each stage
			There are ceil(total_wells/wells_per_stage) total stages

		min_patches: int
			patches to be taken from largest class
			for other classes, the number of patches will be scaled inversely
			with class size to ensure an balanced number of patches in training

		batch_size: int
			Patches in each batch (weight update) during training

	Returns:
		well_df: pd.DataFrame
			Original well_df input with new stage information

		final_patch_df: pd.DataFrame
			Hierarchically sorted DataFrame of patches in corresponding order
			to well_df, with stage and batch information

		class_patch_dist: list of ints
			Distribution of number of patches taken from each class
			with min_patches for largest class and more from others
	'''
	total_wells = len(well_df)
	num_stages = np.ceil(total_wells/wells_per_stage).astype(int)
	classes = well_df['Class'].unique()

	wells_per_class = []
	for i in range(len(classes)):
		wells_per_class.append(len(well_df[well_df['Class']==classes[i]]))
	
	class_patch_dist = get_patches_per_class(min_patches,wells_per_class)

	well_df = well_df.sort_values(by=['Class','Plate'])
	well_df.drop(well_df.columns[well_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
	well_df.reset_index(drop=True,inplace=True)

	# select wells for each stage
	stage_labels = (list(np.arange(num_stages))*wells_per_stage)[:len(well_df)]
	well_df['Stage'] = stage_labels
	well_df.sort_values(by=['Stage','Class','Plate'],inplace=True)
	well_df.reset_index(drop=True,inplace=True)
	patch_df = pd.DataFrame(columns=['Location','Filename','Plate','Class','Stage','Patch'])

	# construct dataframe with patch numbers
	for i in range(num_stages):
		stage_df = well_df[well_df['Stage']==i]

		for j in range(len(classes)):
			class_stage_df = stage_df[stage_df['Class']==classes[j]]
			for w in range(len(class_stage_df)):
				single_patch_df = pd.DataFrame(columns=patch_df.columns)

				single_patch_df['Patch'] = np.arange(class_patch_dist[j])
				single_patch_df[['Location','Filename','Plate','Class','Stage']] = class_stage_df[['Location','Filename','Plate','Class','Stage']].iloc[w].values
				
				patch_df= pd.concat([patch_df,single_patch_df],ignore_index=True)

	# sort into batches
	final_patch_df = pd.DataFrame(columns=list(patch_df.columns)+['Batch'])
	starting_batch = 0
	for i in range(num_stages):
		stage_df = patch_df[patch_df['Stage']==i].copy()
		num_batches,unused_patches = int(len(stage_df)//batch_size),len(stage_df)%batch_size
		if unused_patches > 0:
			stage_df['Batch'] = list(np.arange(starting_batch,num_batches+starting_batch).astype(int))*batch_size + [-1]*unused_patches
		else:
			stage_df['Batch'] = list(np.arange(starting_batch,num_batches+starting_batch).astype(int))*batch_size

		final_patch_df = pd.concat([final_patch_df,stage_df],ignore_index=True)
		starting_batch = starting_batch + num_batches

	final_patch_df.sort_values(by=['Stage','Class','Plate','Filename','Patch'],ignore_index=True,inplace=True)	
	return well_df, final_patch_df, class_patch_dist




class TrainDataGenerator(Sequence):
	'''
	Generates patches for training FocA model

	Attributes:

		df: pd.DataFrame
			table with columns ['Location','Filename','Class'] corresponding to the training set's well-level images

		batch_size: int
			number of wells to include in each batch, number of patches per batch = batch_size * patches_per_well
		
		patch_size: tuple
			dimension of patches, (rows,cols,channels)

		patches_per_well: int
			number of patches to take per well
		
		use_global_normalization:
			if True, normalize patches using NORM_RANGE defined in config.py
			if False, normalize patches at well-level
		
		wells_per_stage: int
			number of well images to be loaded into memory 

		ROI: int
			new dimension of image after being cropped inside well, image size is ROIxROI
		
		FFC: string
			filename of saved flat-field correction image (.npy)
		
		scale: None, float, int
			if < 1, resizes image to size (scale*img.shape[0],scale*img.shape[0])
			if > 1, resizes image to size (scale, scale)

	'''
	batch_size = 10
	patch_size = (100,100,1)
	use_global_normalization = True
	wells_per_stage = 60
	patches_per_well = 25
	ROI = 992
	scale = None
	FFC = None

	def __init__(self,df,batch_size,patch_size,patches_per_well,wells_per_stage=30,use_global_normalization=True,
				FFC=None,scale=None,seed=None):
		self.well_df, self.patch_df,self.class_patch_distribution = make_patch_df(df,wells_per_stage,patches_per_well,batch_size)
		self.num_classes = len(self.class_patch_distribution)
		self.patch_size = patch_size
		self.patches_per_well = patches_per_well
		self.use_global_normalization = use_global_normalization
		self.scale = scale
		if FFC is not None:
			self.FFC = os.path.join(FFC_DIR,FFC)
		else:
			self.FFC = FFC
		self.ROI = np.floor(scale/2).astype(int)
		self.seed = seed
		self.current_stage = 0
		self.loaded_patches = None
		self.loaded_labels = None

	def __len__(self):
		return max(self.patch_df['Batch'].tolist())

	def load_stage(self,stage):
		'''
		Preprocesses a batch of self.wells_per_stage wells and loads their patches into memory

		Args:
			stage: int
				Group of images marked in self.well_df to be loaded together 
		'''
		stage_wells = self.well_df[self.well_df['Stage']==stage]

		for i in range(len(stage_wells)):
			well_info = dict(stage_wells.iloc[i])
			# print(os.path.join(well_info['Location'],well_info['Filename']))
			img = cv.imread(os.path.join(well_info['Location'],well_info['Filename']),
							cv.IMREAD_GRAYSCALE).astype(np.float32)

			if self.scale is not None:
				if self.scale > 1:
					img = resize(img,new_size=(self.scale,self.scale))
				else:
					img = resize(img,scale = self.scale)

			if self.ROI is not None:
				img = center_crop(img,ROI=self.ROI)

			if self.FFC is not None:
				# print("flat-field correcting...")
				FFC_img = np.load(self.FFC,allow_pickle=True)
				
				if self.scale is not None:
					if self.scale > 1:
						FFC_img = resize(FFC_img,new_size=(self.scale,self.scale))
					else:
						FFC_img = resize(FFC_img,scale = self.scale)
				if self.ROI is not None:
					FFC_img = center_crop(FFC_img,ROI=self.ROI)
				if self.use_global_normalization:
					FFC_img[FFC_img>NORM_RANGE[0]] = NORM_RANGE[0]
				prev_range = (np.min(img),np.max(img))
				img = img/(FFC_img+np.finfo(float).eps)
				img = (img-np.min(img))/(np.max(img)-np.min(img))*(prev_range[1]-prev_range[0])+prev_range[0]
				# if stage==1 and i==10:
			# plt.imshow(img)
			# plt.savefig('assets/sample_FFC.png')
			patches = make_patches(img,self.patch_size,self.class_patch_distribution[well_info['Class']],
									use_global_normalization=self.use_global_normalization,
									randomize_location=False,seed=self.seed)

			# patches = make_patches(img,self.patch_size,self.class_patch_distribution[well_info['Class']],
			# 						use_global_normalization=self.use_global_normalization,
			# 						randomize_location=False,augment=True,seed=self.seed)
			
			labels = np.full(len(patches),well_info['Class'])

			if i == 0:
				stage_patches = patches
				stage_labels = labels
			else:
				stage_patches = np.concatenate((stage_patches,patches))
				stage_labels = np.concatenate((stage_labels,labels))
			
		stage_patches = tf.expand_dims(stage_patches,axis=-1)
		self.loaded_patches,self.loaded_labels = stage_patches,stage_labels


	def __getitem__(self,idx):
		
		# CHECK STAGE
		batch_df = self.patch_df[self.patch_df['Batch']==idx]
		batch_stage = batch_df['Stage'].unique()[0]
		if batch_stage != self.current_stage or self.loaded_patches is None:
			self.load_stage(batch_stage)
		self.current_stage = batch_stage
		
		patch_ids = np.array(batch_df.index.tolist()) 
		batch_ids = patch_ids - len(self.patch_df[self.patch_df['Stage'] < batch_stage])

		batch_patches, batch_labels = tf.gather(self.loaded_patches,indices=batch_ids,axis=0),tf.gather(self.loaded_labels,indices=batch_ids,axis=0)

		scrambled_ids = np.random.permutation(len(batch_patches))
		batch_patches,batch_labels = tf.gather(batch_patches,indices=scrambled_ids),tf.gather(batch_labels,indices=scrambled_ids)
		return batch_patches, batch_labels
	


class TestDataGenerator(Sequence):
	'''
	Generates patches for the validation and testing steps of training GoogleMIQ model

	Attributes:
		
		df: pd.DataFrame
			table with columns ['Location','Filename','Class'] corresponding to the validation set's well-level images


		wells_per_batch: int
			number of well images in each batch (e.g., test_datagen[i])
			patches per batch will be patches()
			
		patch_size: tuple
			dimension of patches, (rows,cols,channels)

		patches_per_well: int
			number of patches to take per well
		
		use_global_normalization:
			if True, normalize patches using NORM_RANGE defined in config.py
			if False, normalize patches at well-level

		ROI: int
			new dimension of image after being cropped inside well, image size is ROIxROI
		
		FFC: string
			filename of saved flat-field correction image (.npy)
		
		scale: None, float, int
			if < 1, resizes image to size (scale*img.shape[0],scale*img.shape[0])
			if > 1, resizes image to size (scale, scale)

		randomize_patch_order: bool
			if True, randomize the order of patches within each batch
			default: False

	'''
	df = None
	wells_per_batch = 30
	patch_size = (100,100)
	patches_per_well = 16
	use_global_normalization = True
	ROI = 992
	FFC = None
	scale = None
	current_batch = None

	def __init__(self,df,wells_per_batch,patch_size,patches_per_well,use_global_normalization=True,FFC=None,scale=None):
		self.df = df
		self.classes = list(df['Class'].unique())
		self.wells_per_batch = wells_per_batch
		self.patch_size = patch_size
		self.patches_per_well = patches_per_well
		self.ROI = np.floor(scale/2).astype(int)
		self.use_global_normalization = use_global_normalization
		self.scale = scale
		if FFC is not None:
			self.FFC = os.path.join(FFC_DIR,FFC)
		else:
			self.FFC = FFC
		self.current_batch = None

	def __len__(self):
		return math.ceil(len(self.df)/float(self.wells_per_batch))

	def __getitem__(self,idx):
		batch_df = self.df.iloc[idx*self.wells_per_batch:(idx+1)*self.wells_per_batch]
		self.current_batch = batch_df

		for i in range(len(batch_df)):
			well_info = batch_df.iloc[i]
			img = cv.imread(os.path.join(well_info['Location'],well_info['Filename']),
							cv.IMREAD_GRAYSCALE).astype(np.float32)

			if self.scale is not None:
				if self.scale > 1:
					img = resize(img,new_size=(self.scale,self.scale))
				else:
					img = resize(img,scale = self.scale)
			
			if self.ROI is not None:
				img = center_crop(img,ROI=self.ROI)


			if self.FFC is not None:
				FFC_img = np.load(self.FFC,allow_pickle=True)
				
				if self.scale is not None:
					if self.scale > 1:
						FFC_img = resize(FFC_img,new_size=(self.scale,self.scale))
					else:
						FFC_img = resize(FFC_img,scale = self.scale)
				if self.ROI is not None:
					FFC_img = center_crop(FFC_img,ROI=self.ROI)
				if self.use_global_normalization:
					FFC_img[FFC_img>NORM_RANGE[0]] = NORM_RANGE[0]
				prev_range = (np.min(img),np.max(img))
				img = img/(FFC_img+np.finfo(float).eps)
				img = (img-np.min(img))/(np.max(img)-np.min(img))*(prev_range[1]-prev_range[0])+prev_range[0]

			patches = make_patches(img,self.patch_size,self.patches_per_well,use_global_normalization=self.use_global_normalization)
			labels = np.full(len(patches),well_info['Class'])

			if i == 0:
				batch_patches = patches
				batch_labels = labels
			else:
				batch_patches = np.concatenate((batch_patches,patches))
				batch_labels = np.concatenate((batch_labels,labels))

		batch_patches = tf.expand_dims(batch_patches,axis=-1)
		return batch_patches, batch_labels

# def main():
# 	train_df = pd.read_csv(os.path.join(DATASET_CSV_DIR,'deployment_heldout.csv'))
# 	# _,dl_config, _ = load_config('default.yaml')
# 	datagen = TestDataGenerator(train_df,wells_per_batch=32,patch_size=(100,100,1),patches_per_well=16,scale=1562)
# 	for i in range(1,2):
# 		# print(i)
# 		batch_patches, batch_labels = datagen[i]
# 		# print(batch_labels)
# 	import matplotlib.pyplot as plt
# 	fig,ax = plt.subplots(nrows=4,ncols=8,figsize=(12,6))
# 	for i in range(32):
# 		ax[i//8,i%8].imshow(batch_patches[i+272],vmin=0,vmax=1)
# 		ax[i//8,i%8].axis('off')
# 	plt.show()
# # 	# well_df, patch_df = make_patch_df(train_df, 30, 16,2)
# # 	# print(patch_df)
# # 	# patch_df.to_csv('/mnt/data-storage/FocusBotData/sample_patch_df.csv')
# if __name__ == "__main__":
# 	main()
# def order_df_from_batch_sizes(df,batch_dist,classes):
# 	ordered_df = pd.DataFrame(columns=df.columns)
# 	class_dfs = []
# 	for c in range(len(classes)):
# 		class_dfs.append(df[df['Class']==classes[c]].reset_index(drop=True))
	
# 	# for c in range(len(class_dfs)):
# 	# 	wells = class_dfs[c]['Filename']
# 	# 	plate_col = []
# 	# 	for well in wells:
# 	# 		plate_col.append('_'.join(well.split('_')[0:3]))
# 	# 	class_df = class_dfs[c]
# 	# 	class_df.loc[:,'Plate'] = plate_col
# 	# 	class_dfs[c] = class_df
# 	plate_indices = []
# 	for c in range(len(classes)):
# 		class_plates = list(class_dfs[c]['Plate'].unique())
# 		class_plate_indices = []
# 		for p in range(len(class_plates)):
# 			class_plate_indices.append(class_dfs[c].index[class_dfs[c]['Plate']==class_plates[p]])
# 		plate_indices.append(class_plate_indices)

# 	batch_begin = np.zeros(len(classes))
# 	for i in range(len(batch_dist)):
# 		batch_end = batch_begin + batch_dist[i]
# 		for c in range(len(classes)):
# 			class_plate_indices = plate_indices[c]
# 			for p in range(int(batch_dist[i][c])):
# 				plate_ids = list(class_plate_indices[p%len(class_plate_indices)])
# 				row_id = plate_ids.pop()
# 				if plate_ids == []:
# 					class_plate_indices.pop(p%len(class_plate_indices))
# 				else:
# 					class_plate_indices[p%len(class_plate_indices)] = plate_ids
# 				# print(len(class_dfs[c]))
# 				# print(class_dfs[c].iloc[row_id])
# 				ordered_df = pd.concat([ordered_df,class_dfs[c].iloc[[row_id]]])
# 			# ordered_df = pd.concat([ordered_df,class_dfs[c][int(batch_begin[c]):int(batch_end[c])]])
# 		batch_begin = batch_end
# 	return ordered_df

# def distribute_classes_across_batches(df,batch_size=60):
# 	'''
# 	Distribute classes evenly across batches, and determine optimal number of patches (per well) per class to offset class imbalance
# 	Redistribute wells across other batches if size of training set is not divisible by batch_size

# 	Args:
# 		df: pd.DataFrame

# 		batch_size: int
	
# 	Returns:


# 	'''
# 	classes = df['Class'].unique()
# 	num_batches = math.ceil(len(df)/float(batch_size))
# 	class_dist = {}	
# 	for i in range(len(classes)):
# 		class_dist[classes[i]] = len(df[df['Class']==classes[i]])
	
# 	wells_per_batch = {}
# 	for i in range(len(classes)):
# 		wells_per_batch[classes[i]] = math.floor(class_dist[classes[i]]/num_batches)

# 	redistribute = False
# 	remainders = []
# 	for i in range(len(classes)):
# 		remainder = class_dist[classes[i]]%num_batches
# 		remainders.append(remainder)
# 		if remainder > 0:
# 			redistribute = True

# 	if redistribute:
# 		wpb_array = np.zeros((num_batches,len(classes)))
# 		for c in range(len(classes)):
# 			r = remainders[c]
# 			for i in range(len(wpb_array)):
# 				if r <=0:
# 					continue
# 				wpb_array[i,c] +=1
# 				r-=1

# 			wpb_array[:,c] += wells_per_batch[c]

# 	else:
# 		wpb_array = np.zeros((num_batches,len(classes)))
# 		for c in range(len(classes)):
# 			wpb_array[:,c] = wells_per_batch[c]
# 	return order_df_from_batch_sizes(df,wpb_array,classes),wpb_array