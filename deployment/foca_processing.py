'''
Functions for the intermediate processing between predicting focus levels of wells and
ascertaining/outputting focal quality of plates
'''
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
import gc
import os
from datetime import datetime

import foca_analysis as foca_analysis

import pandas as pd

#from slack_sdk import WebClient

from input.config import CSV_OUTPUT, MODEL_WEIGHTS, DATA_RESOLUTION
# from input.credentials.keys import SLACK_TOKEN ## Uncomment if you've set up for the Slack features
from models.foca import FocA
from google.cloud import storage
import inquirer
gc.collect()
def yesno(question):
    prompt = f'{question} (y/n): '
    answer = input(prompt).strip().lower()

    if answer not in ['y', 'n']:
        print(f'{answer} is an invalid choice, please select y/n')
        return yesno(question)

    if answer == 'y':
        return True

    return False
# Loading model into memory
if not os.path.exists(MODEL_WEIGHTS):
	use_pretrained = yesno("ERROR: MODEL_WEIGHTS are invalid. Would you like to download and use pre-trained weights?")
	if use_pretrained:

		print('Downloading pretrained weights...')
	
		# download weights from Google Cloud
		parent_dir = os.path.join(str(Path(__file__).parents[1]),'input')
		storage_client = storage.Client.create_anonymous_client()
		bucket_name = "foca2023"
		bucket = storage_client.bucket(bucket_name)
		blobs=bucket.list_blobs()
		blobs = [blob for blob in blobs if blob.name.startswith('pretrained_weights') and not blob.name.endswith('/')]

		for blob in blobs:
			blob = bucket.blob(blob.name)
			blob_path =  "/".join(blob.name.split('/')[:-1])
			if not os.path.isdir(os.path.join(parent_dir,blob_path)):
				path = Path(os.path.join(parent_dir,blob_path))
				path.mkdir(parents=True)
			blob.download_to_filename(os.path.join(parent_dir,blob.name))
		MODEL_WEIGHTS = os.path.join(parent_dir,"pretrained_weights/final_config")

	else:
		[parent_dir,desired_weights]="/".join(MODEL_WEIGHTS.split('/')[:-2]),MODEL_WEIGHTS.split('/')[-1]
		other_weights = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir,d)) and d != desired_weights]
		if len(other_weights)>0:

			choose_other_weights = yesno("Would you like to choose some other weights?")
			question = [
				inquirer.List('weights',
								message='Which weights would you like to use?',
								choices=other_weights,
							 ),
						]
			weights = inquirer.prompt(question)['weights']
			MODEL_WEIGHTS = os.path.join(parent_dir,weights)
		else:
			
			exit()
else:
	pass

foca = FocA(weights=MODEL_WEIGHTS)

# Connecting to slack
# slack_client = WebClient(token=SLACK_TOKEN) ## Uncomment if you've set up for the Slack features

def check_processed_plates() :
	if (os.path.exists(CSV_OUTPUT)) and os.stat(CSV_OUTPUT).st_size != 0 :
		processed_plates = pd.read_csv (CSV_OUTPUT)
		processed_plates = processed_plates[["RUN","PLATE","SCAN_TIME"]]
		processed_plates["NAME"] = processed_plates["RUN"] + "_" + processed_plates["PLATE"] + processed_plates["SCAN_TIME"] 
		
	else :
		processed_plates = pd.DataFrame(columns=['NAME'])
		open(CSV_OUTPUT, 'w')
	processed_plates = processed_plates["NAME"].tolist()
		
	return processed_plates

def process_active_plates(directory,run_type=None,overwrite=False,device='/device:CPU:0'):
	'''
	Get focus information for all available plate scans

	Args:
		directory: string
			Directory where all run directories are stored

		run_type: string
			If desired, run prefix to filter run directories 
		
		overwrite: bool
			process and overwrite already processed scans

	Returns:
		all_run_info: dictionary
			Dictionary with hierarchy:
				Run -> Plate -> Well
			where wells have an is_OOF binary int (1: Out-of-Focus, 0: In-Focus)
	'''
	runs = [f.name for f in os.scandir(directory)if f.is_dir()]
	if run_type is not None:
		runs = [f for f in runs if f.startswith(run_type)]
	runs = sorted(runs)

	all_run_info = {}
	for run in runs:
		run_dict = process_run(run,directory=directory,overwrite=overwrite,device=device)
		all_run_info.update(run_dict)
	return all_run_info

def process_run(run,directory,overwrite=False,device='/device:CPU:0'):
	'''
	Get focus information for all plates in a run

	Args:
		run: string
			Name of run to analyze

		directory: string
			Name of directory where plate directories for desired run are 

		overwrite: bool
			process and overwrite already processed scans
	
	Returns:
		run_dict: dictionary
			Dictionary with hierarchy:
				Run -> Plate -> Well
			where wells have an is_OOF binary int (1: Out-of-Focus, 0: In-Focus)
	'''
	run_dir = os.path.join(directory,run)
	plates_dict = {}
	try:
		plates = sorted(list(next(os.walk(run_dir))[1]))

		if len(plates)>0:
			for i,_ in enumerate(plates):
				plate_dict = process_plate(run,plates[i],directory,overwrite=overwrite,device=device)
				if plate_dict == None :
					continue
				else: 
					plates_dict.update(plate_dict)
			run_dict = {run:plates_dict}
			return run_dict
	except StopIteration:
		print('Skipping run %s -- No plates in run'%run)
		return {run:plates_dict}
	
def process_plate(run,plate,directory, single_plate=False,overwrite=False,device='/device:CPU:0'):
	'''
	Get focus information for a given plate

	Args:
		run: string
			Name of run to analyze

		plate: string
			Name of plate to analyze

		directory: string
			Directory where a plate's exports are stored (subdirectories should be 'TIFF01','TIFF04',etc)


		single_plate: bool
			Flag for if plate is the only one being analyzed in a given run
		
		overwrite: bool
			process and overwrite already processed scans
			
	Returns:
		run_dict: dictionary
			Dictionary with hierarchy:
				Run -> Plate -> Well
			where wells have an is_OOF binary int (1: Out-of-Focus, 0: In-Focus)
			Excludes return of plate_dict

		plate_dict: dictionary
			Dictionary with hierarchy:
				Plate -> Well
			where wells have an is_OOF binary int (1: Out-of-Focus, 0: In-Focus)
			Excludes return of run_dict

	'''
	if single_plate:
		run_dict = {}
		plate = "_".join([run,plate])
	plate_dir = os.path.join(directory,run,plate)
	try:
		scales = list(next(os.walk(plate_dir))[1])
		
		tiff_dir = 'TIFF' + DATA_RESOLUTION.split('um')[0].zfill(2)
		if (tiff_dir in scales):
			scale_dir = os.path.join(plate_dir,tiff_dir)
			scans = sorted(list(next(os.walk(scale_dir))[1]))
			image_dir = os.path.join(scale_dir,scans[0])
			processed_plates = check_processed_plates() 
			scan_time_str =image_dir.split('/')[-1].split('_')[-1]
			scan_time = datetime.strptime(scan_time_str,'%m-%d-%Y-%I-%M-%S-%p')
			scan_time = f'{scan_time}'
			scan_str = plate + scan_time
			plate_name = "_".join(plate.split('_')[1:])
			if scan_str not in processed_plates or overwrite==True:
				print('Processing plate %s...'%plate)
				focus_dict,plate_scan_time = foca_analysis.analyze_plate(plate,image_dir,foca,device=device)
				print('%d wells out of focus'%list(focus_dict.values()).count(1))
				# plate_name = "_".join(plate.split('_')[1:])
				if single_plate:
					run_dict[run] = {plate_name:(focus_dict,plate_scan_time)}
					return run_dict
				else:
					plate_dict = {plate_name:(focus_dict,plate_scan_time)}
					return plate_dict
			else :
				print('Skipping plate %s -- Plate already processed '% plate)
				plate_dict = {plate_name:({},0)}
		else:
			print('Skipping plate %s -- Export scale unavailable '% plate)
			plate_name = "_".join(plate.split('_')[1:])
			if single_plate:
				run_dict[run] = {plate_name:({},0)}
				return run_dict
			else:
				plate_dict = {plate_name:({},0)}
				return plate_dict
	except StopIteration:
		print('Skipping plate %s -- Plate does not exist'%plate)
		return run_dict

