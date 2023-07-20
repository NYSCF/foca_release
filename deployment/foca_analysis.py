import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
import os
import re
from datetime import datetime

import foca_utils as foca_utils
import numpy as np
import pandas as pd
from dataloader import DeploymentDataGenerator
from foca_utils import save_image, visualize_focal_classes

from input.config import (FLAGGED_DIRS, FOCUS_VISUALIZATION_DIR,
                          PATCH_SELECTION_METHOD, SORT_IMAGES)
from models.foca import FocA
import tensorflow as tf
# DataGenerator parameters
PATCHES_PER_WELL = 12
PATCH_SIZE = (100,100,1)

def get_datagen(test_dir,use_center_patches=False,batch_size=48):
	'''
	Generate DataGenerator that produces patches for well-level focal classification

	Args:
		test_dir: string
			Path where well-level images are stored
		
		use_center_patches: bool
			Flag to get patches from subgrid centered around middle of well
	
	Returns: 
		datagen: DeploymentDataGenerator object
			Datagenerator to feed patches into our deep learning classification model
	'''
	files = [f for f in os.listdir(test_dir) if f.endswith('.tiff') or f.endswith('.jpg') or f.endswith('.png')]
	files = [f for f in files if re.search('Ch2',f) is not None]
	wells = []
	for i in files:
		well = i.split('_')[1]
		wells.append(''.join([well[0],well[1:].zfill(2)]))
	ids = list(np.argsort(wells))
	df = pd.DataFrame(columns=['Location','Filename'])
	df['Filename'] = list(np.array(files)[ids])
	df['Location'] = test_dir
	# df = df.sort_values(by='Filename')
	datagen = DeploymentDataGenerator(df,batch_size=batch_size,patches_per_well=PATCHES_PER_WELL,patch_size=PATCH_SIZE,
				   use_center_patches=use_center_patches,FFC=None,patch_selection_method=PATCH_SELECTION_METHOD)
	return datagen

def assess_focus(results,weight_by_confidence=True):
	'''
	Converts array of raw model prediction scores into binary in-focus/out-of-focus class

	Args:
		results: np.ndarray
			patch-level focal class scores of shape (3, num_patches)
	
	Returns:
		is_OOF: int
			1 if well is out-of-focus, 0 if in-focus	
	'''
	if weight_by_confidence:
		confidence = np.absolute(results-0.5)*2
		normalized_confidence = confidence/np.sum(confidence)
		class_preds = np.sum(normalized_confidence*results)
	else:
		class_preds = np.mean(results,axis=0)
	
	if class_preds <0.5:
		# in-focus
		# print(class_preds)
		is_OOF = 0
	else:
		# out-of-focus
		is_OOF = 1

		# print(class_preds)

	return is_OOF

def analyze_plate(plate,image_dir,model,save_viz=True,batch_size=96,device='/device:CPU:0'):
	'''
	1. Iterates over all wells in a plate and gets focus level of each
	2. Outputs a visualization of focal quality as plate map

	Args:
		plate: string
			Plate barcode

		image_dir: string
			Path of directory where a plate's images are stored

		model: FocA object
			Deep learning model to evaluate focal quality of well patches

		save_viz: bool
			If True, save a plate_map visualization
		
		batch_size: int
			Number of wells for model to predict at one time

	Returns:
		focus_dict: dictionary

		scan_time: datetime
	'''
	datagen = get_datagen(image_dir,use_center_patches=False,batch_size=batch_size)
	scan_time_str =image_dir.split('/')[-1].split('_')[-1]
	scan_time = datetime.strptime(scan_time_str,'%m-%d-%Y-%I-%M-%S-%p')
	focus_dict = {}


	# Mark empty wells as empty in focus dict
	# KEY: 0 = in-focus, 1 = out-of-focus, 2 = empty
	# You can manually insert this information by providing the indexes of the empty wells here:
	#   empty_wells=[]

	# for i in empty_wells:
	# 	focus_dict[i] = 2

	# Loop over batches of wells to analyse
	with tf.device(device):
		for i in datagen:
			results = model.model(i)
			batch_df = datagen.current_batch
			
			# Loop over patch classification results for each well
			for k in range(round(len(results)/PATCHES_PER_WELL)):
				# print(k)
				well_results = results[k*PATCHES_PER_WELL:(k+1)*PATCHES_PER_WELL]
				fname = batch_df['Filename'].iloc[k]
				img_dir = batch_df['Location'].iloc[k]
				
				well = fname.split('_')[1]
				well = well[0] + well[1:].zfill(2)

				# Checking if well is already marked empty
				if well not in list(focus_dict.keys()):
					is_OOF = assess_focus(well_results)

					focus_dict[well] = is_OOF
					
					# Option to sort images for retraining later
					if SORT_IMAGES:
						[file_name,file_ext] = fname.split('.')
						retraining_fname = ".".join(["_".join([file_name,scan_time_str]),file_ext])
						save_image(img_dir,fname,save_name=retraining_fname,save_dir=FLAGGED_DIRS[is_OOF])
				else:
					pass

	# Produce plate-level visualization for Slack
	if not os.path.exists(FOCUS_VISUALIZATION_DIR):
		try:
			os.makedirs(FOCUS_VISUALIZATION_DIR)
		except Exception:
			pass
	if save_viz and os.path.exists(FOCUS_VISUALIZATION_DIR) and any(well == 1 for well in list(focus_dict.values())):
		visualize_focal_classes(focus_dict,plate,save_dir=FOCUS_VISUALIZATION_DIR)
	return focus_dict,scan_time

