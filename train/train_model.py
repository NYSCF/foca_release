import sys
from pathlib import Path

import tensorflow as tf
from dataloader import TestDataGenerator, TrainDataGenerator
from dataset import create_datasets
from model_generation import generate_trained_model, load_config

sys.path.append(str(Path(__file__).parents[1]))
import argparse
import os

import pandas as pd

from input.config import *


def yesno(question):
    prompt = f'{question} (y/n): '
    answer = input(prompt).strip().lower()

    if answer not in ['y', 'n']:
        print(f'{answer} is an invalid choice, please select y/n')
        return yesno(question)

    if answer == 'y':
        return True

    return False

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--name',
						help="name of directory in which to save model",
						default="foca_weights",type=str)
	parser.add_argument('-y','--yaml',
		     			help='YAML file with config to use during training',
						default='default_training.yaml',type=str)
	parser.add_argument('-d','--device',
		     			help='Use CPU or GPU, valid inputs are [CPU, GPU]',
						default='GPU',type=str)
	parser.add_argument('--verbose',action='store_true',
		     			help='If True, Output details of samples trained')
	parser.add_argument('--generic',action='store_true',
		     			help='If True, assume data is formatted generically and avoid plate variety optimization')
	args = vars(parser.parse_args())

	
	
	model_path = os.path.join(WEIGHT_DIR,args['name'])
	
	# Check if model already exists
	if not os.path.isdir(model_path):
		os.makedirs(model_path)
	else:
		question = "%s already exists, do you want to overwrite it?"%model_path
		overwrite = yesno(question)
		if not overwrite:
			question = "Continue with a new name?"
			continue_training = yesno(question)
			if continue_training:
				args['name'] = input("Enter a new model name: ")
				model_path = os.path.join(WEIGHT_DIR,args['name'])
				if not os.path.exists(model_path):
					os.makedirs(model_path)

			else:
				return 0
		
		
	print('Generating trained model to save in directory %s/'%os.path.join(WEIGHT_DIR, args['name']))

	# Generate training and validation sets
	training_csv,validation_csv = create_datasets(DATA_DIR,DATA_CLASS_DIRS,folds=5,max_class_size=MAX_CLASS_SIZE,csv_prefixes=('train','val'),generic=args['generic'])
	
	training_set = os.path.join(DATASET_CSV_DIR,training_csv)
	validation_set = os.path.join(DATASET_CSV_DIR, validation_csv)

	model_config, train_dataloader_config, val_dataloader_config, _ = load_config(args['yaml'])

	# Choosing device based on availability and user input
	devices = tf.config.list_logical_devices()
	if args['device']=='GPU' and len(devices) > 1:
		print('Computing on GPU (%s)'%devices[1].name)
		train_device = devices[1].name
	else:
		print('Computing on CPU (%s)'%devices[0].name)
		train_device = devices[0].name

	# Train the model
	with tf.device(train_device):
		train_df = pd.read_csv(training_set)
		train_datagen = TrainDataGenerator(train_df,**train_dataloader_config)

		val_df = pd.read_csv(validation_set)
		val_datagen = TestDataGenerator(val_df,**val_dataloader_config)

		generate_trained_model(train_datagen,model_config, val_datagen=val_datagen, 
			 					save_dir=model_path, verbose=args['verbose'])


if __name__=="__main__":
	main()
