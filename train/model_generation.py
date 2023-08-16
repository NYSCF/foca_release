'''
FocA model training and hyperparameter optimization functions
'''

import gc
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
import math

import yaml
from tensorflow.python.keras.callbacks import LearningRateScheduler

from input.config import CONFIG_DIR, SCALE_DICT, WEIGHT_DIR
from models.foca import FocA


def lr_time_decay(epoch,lr,decay=0.1):
	'''
	Learning rate scheduling based on time at a fixed decay rate
	'''
	
	return lr * 1 / (1 + decay * epoch)

def lr_exp_decay(epoch, initial_learning_rate=0.005*10):
	'''
	Learning rate scheduling with exponential decay
	'''
	k = 0.01
	return initial_learning_rate * math.exp(-k*epoch)

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
	with open(os.path.join(CONFIG_DIR,config_file)) as f:
		config = yaml.load(f.read(),Loader=yaml.CLoader)
	f.close()

	model_config,dataloader_config,dataloader_test_config = config['model'], config['dataloader'],config['dataloader_test']
	#scale learning rate with batch_size
	model_config['lrn_rate'] = model_config['lrn_rate']*dataloader_config['batch_size']
	scale_name = dataloader_config.pop('scale')
	dataloader_test_config.pop('scale')
	if type(scale_name) is not list:
		dataloader_config.update(scale=SCALE_DICT[scale_name])
		dataloader_test_config.update(scale=SCALE_DICT[scale_name])
	return model_config, dataloader_config, dataloader_test_config, scale_name


def generate_trained_model(datagen,model_config,val_datagen=None,save_dir=None,verbose=True):
	'''
	Generate a trained classification model given a training set and
	configuration
	
	Args:
		datagen: TrainDataGenerator
			data generator training set

		model_config: dict
			dictionary with model configuration parameters

		val_datagen: ValidationDataGenerator
			data generator for validation set
			

		save_dir: string
			directory where to save model weights
	
	Return:
		foca: FocA
			Focus analysis deep learning model
	'''
	
	gc.collect()
	foca = FocA(**model_config)
	early_stop = foca.early_stop
	if val_datagen is not None:
		foca.model.fit(datagen,epochs=30,batch_size=1,verbose=True,validation_data=val_datagen,
					callbacks=[LearningRateScheduler(lr_time_decay),early_stop],shuffle=False)
	else:
		foca.model.fit(datagen,epochs=30,batch_size=1,verbose=True,shuffle=False)
	
	if save_dir is not None:
		model_path = save_dir#os.path.join(WEIGHT_DIR,save_dir)
		if verbose:
			print("Model saved to %s"%model_path)
		foca.model.save(model_path+'/')
	return foca

# for i in range(len(datagen)):
	# 	patches, labels = datagen[i]
	# 	print("Batch %d out of %d"%(i+1,len(datagen)))
	# 	foca.model.train_on_batch(patches,labels)
	# 	if verbose:
	# 		print(datagen.current_batch)
	# 	if val_datagen is not None:
	# 		foca.model.fit(patches,labels,epochs=1,batch_size=32,verbose=True,shuffle=False,validation_data=(val_patches,val_labels),callbacks=[early_stop])
	# 	else:
	# 		foca.model.fit(patches,labels,epochs=1,batch_size=32,verbose=True,shuffle=False)
	# 	del patches
	# 	del labels
