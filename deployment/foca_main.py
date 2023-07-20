'''
Main file called to run FocA in a deployment setting
'''
import argparse
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf
from input.config import IMAGE_DIR
from foca_output import (dict_to_df, generate_slack_message, notify_slack,
                         update_csv)
from foca_processing import process_active_plates, process_plate, process_run
#from foca_utils import get_db_cursor, get_empty_wells
#from input.credentials.keys import SLACK_TOKEN ## Uncomment if you've set up for the Slack features
from input.config import MODEL_WEIGHTS
from slack_sdk import WebClient

#slack_client = WebClient(token=SLACK_TOKEN) ## Uncomment if you've set up for the Slack features

print('\nFocus Analysis Started')
print('----------------------')
def FocAMain():

	# Allow optional parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--run',
						help="run ID if only one run is to be processed",
						default="all")
	parser.add_argument('-rt','--run_type',
						help='run type which is the prefix of a run (e.g., MON, RCON,...)',
						default=None)
	parser.add_argument('-p', '--plate',
						help="plate ID if only one plate is to be processed",
						default="all")
	parser.add_argument('-d', '--dir',
						help='path to directory if different from celigo exports',
						default='ce')
	parser.add_argument('-fd', '--flagged_dir',
						help='path to directory to store images flagged OOF',
						default='')
	parser.add_argument('-dev','--device',
		     			help='specify device type as CPU or GPU, or device-type and number (e.g., GPU_0 for the first GPU)',
						default='GPU_0')
	parser.add_argument('--overwrite_scans', action='store_true',
						help="flag for reprocessing already processed scans")
	args = vars(parser.parse_args())
	
	
	# Determine available GPUs and CPUs to compute on
	devices = tf.config.list_logical_devices()
	CPUs = [d for d in devices if d.device_type=='CPU']
	GPUs = [d for d in devices if d.device_type=='GPU']
	device_type = args['device'][:3]
	if len(args['device'])>3:
		device_num = int(args['device'][4:])
	else:
		device_num = 0
	if len(GPUs)>0 and device_type =='GPU':
		print("%d GPU(s) available"%len(GPUs))
		try:
			device = GPUs[device_num]
			print('Using GPU %d'%device_num)
		except IndexError:
			device = GPUs[0]
			print('Using GPU %d'%0)


	elif len(CPUs)>0:
		try:
			device = GPUs[device_num]
			print('Using CPU %d'%device_num)
		except IndexError:
			device = CPUs[0]
			print('Using CPU %d'%0)
	else:
		exit()
	print('----------------------')
	if args['run'] != 'all':
		print('Processing only run ' + args['run'])
		if args['plate'] != 'all':
			print('Processing only plate ' + args['plate'])
			focus_dict = process_plate(args['run'],args['plate'],
													directory=IMAGE_DIR,
													overwrite=args['overwrite_scans'],
													single_plate=True,device=device)
		else:
			focus_dict = process_run(args['run'],directory=IMAGE_DIR,
													overwrite=args['overwrite_scans'],
													device=device)

	else:
		focus_dict = process_active_plates(directory=IMAGE_DIR,
											run_type=args['run_type'],
											overwrite=args['overwrite_scans'],
											device=device)

	focus_df, OOF_plates = dict_to_df(focus_dict)
	
	if focus_df.empty :
		print('No new plates to be processed')
		# notify_slack(text='No new plates to be processed',images=[],channel='testing')	
	else:
		# Update csv
		update_csv(focus_df)

		# Write to slack
		# slack_message = generate_slack_message(focus_df)
		
		# notify_slack(text=slack_message,images=OOF_plates,channel='testing')	
	

	
	
if __name__ == "__main__":
	FocAMain()
	