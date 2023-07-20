import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parents[1]))
import os
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
from slack_sdk import WebClient

from input.config import CSV_OUTPUT,FOCUS_VISUALIZATION_DIR
# from input.credentials.keys import (AWS_ACCESS_KEY, AWS_SECRET_KEY, DATABASE,
#                                     DBPASSWORD, DBUSERNAME, SERVER,
#                                     SLACK_TOKEN) ## Uncomment if you've set up for the Slack features


def dict_to_df(results_dict):
	'''
	Convert dictionary with focal class information into a more organized DataFrame 
	
	Args:
		results_dict: dictionary
			Dictionary with hierarchy:
				Run -> Plate -> Well
			where wells have an is_OOF binary int (1: Out-of-Focus, 0: In-Focus)
				
	Returns:
		results_df: pd.DataFrame
			DataFrame with same information as results_dict + some additional info
			derived from filenames

	'''
	OOF_plates = []
	runs = list(results_dict.keys())
	columns=['RUN','PLATE','SCAN_TIME','IS_OOF','OOF_COUNT','IF_COUNT','EMPTY_COUNT','OOF_WELLS']
	df_data = []
	for run in runs:
		plate_dict = results_dict[run]
		plates = list(plate_dict.keys())
		for plate in plates:
			has_OOF_wells = False
			wells = list(plate_dict[plate][0].keys())
			if len(wells)==0:
				continue
			OOF_wells = []
			empty_count = 0
			for well in wells:
				is_OOF = plate_dict[plate][0][well]
				if is_OOF == 1:
					has_OOF_wells = True
					OOF_plates.append(os.path.join(FOCUS_VISUALIZATION_DIR, run+ '_' + plate + '.png'))
					OOF_wells.append(well)
				elif is_OOF == 2:
					empty_count+=1
			plate_data = [run,plate,plate_dict[plate][1],int(has_OOF_wells),
							len(OOF_wells),len(wells)-len(OOF_wells)-empty_count,empty_count,
							','.join(sorted(OOF_wells))]
			df_data.append(plate_data)
	results_df=pd.DataFrame(data=df_data,columns=columns)	
	OOF_plates = list(set(OOF_plates))
	return results_df, OOF_plates

def generate_slack_message(df):
	'''
	Generate string with a summary of focal analysis for all plates processed
	
	Args:
		df: pd.DataFrame
			Dataframe containing processed plate scans focal analysis information

	Returns:
		slack_message: string
			Slack message in a single string (\n used for newlines)

	'''
	print(df)
	slack_message = 'Focal analysis of all plates is complete.\n\n'
	slack_message +='The following plates were analyzed:\n'
	runs = list(df['RUN'].unique())
	for run in runs:
		plates = list(df[df['RUN'] == run]['PLATE'].unique())
		if len(plates) == 1:
			slack_message += '%s_%s\n'%(run,plates[0])
		else:	
			slack_message += '%s:\n	%s\n'%(run,'\n	'.join(plates))
	
	if 1 in df['IS_OOF'].unique():
		slack_message += 'The following plates appear to be out of focus:\n'
		OOF_plate_df = df[df['IS_OOF'] == 1]
		OOF_runs = sorted(list(OOF_plate_df['RUN'].unique()))
		for oof_run in OOF_runs:
			OOF_plates = OOF_plate_df[OOF_plate_df['RUN'] == oof_run]
			plate_list = sorted(list(OOF_plates['PLATE'].unique()))
			plate_info = []
			for pl in plate_list:
				current_plate = pl
				OOF_plate = OOF_plates[OOF_plates['PLATE'] == current_plate]
				OOF_count = OOF_plate['OOF_COUNT']
				total_count = OOF_plate['OOF_COUNT']+OOF_plate['IF_COUNT']
				plate_info.append(list(OOF_plate['PLATE'])[0] + ' (%d / %d OOF)'%(OOF_count,total_count))

			if len(plate_info)==1:
				slack_message += '%s_%s\n'%(oof_run,plate_info[0])
			else:
				slack_message += '%s:\n	%s\n'%(oof_run,'\n	'.join(plate_info))
		slack_message += '\n'
	else:
		slack_message += '\nAll scans appear to be in focus. If you find an out of focus scan, please notify the channel.\n\n'

	return slack_message

def notify_slack(text,images,channel='testing'):
	'''
	Sends a slack message displaying focal analysis information to a specified channel

	Args:
		text: string
			Slack message in a single string (\n used for newlines)

		channel: string
			Slack channel in which to post message

		images: list
		list of OOF plates images to be slacked to prod team

	'''
	slack_client = WebClient(token=SLACK_TOKEN)
	slack_client.chat_postMessage(channel=channel,text=text)

	for im in images:
		slack_client.files_upload(
    	channels=channel,
    	file=im)

def update_csv(focus_df):
	'''
	Update the CSV with the last scan for each plate, overwriting the last scans
	Uploads the last CSV on AWS to update the DB

	Args:
		focus_df: pd.DataFrame
			Dataframe containing processed plate scans focal analysis information
	'''

	now = datetime.now()
	if os.stat(CSV_OUTPUT).st_size == 0:
			new_df = focus_df
	else :
		df = pd.read_csv(CSV_OUTPUT)
		new_df = pd.merge(df,focus_df,on=['RUN','PLATE'],how='outer')
		new_df = new_df.sort_values(by=['OOF_WELLS_y'], ascending=False)
		# print(new_df)
		for i,_ in new_df.iterrows():

			old_scan_time = new_df.at[i,'SCAN_TIME_x']

			new_scan_time = new_df.at[i,'SCAN_TIME_y']
			if (old_scan_time == '') or (old_scan_time is np.nan):
				old_scan_time = datetime.min
			else:
				old_scan_time = datetime.strptime(old_scan_time,'%Y-%m-%d %H:%M:%S')
			if (new_scan_time > old_scan_time):
				new_df.at[i,'SCAN_TIME'] = new_df.at[i,'SCAN_TIME_y']
				new_df.at[i,'IS_OOF'] = new_df.at[i,'IS_OOF_y']
				new_df.at[i,'IF_COUNT'] = new_df.at[i,'IF_COUNT_y']
				new_df.at[i,'OOF_COUNT'] = new_df.at[i,'OOF_COUNT_y']
				new_df.at[i,'EMPTY_COUNT'] = new_df.at[i,'EMPTY_COUNT_y']
				new_df.at[i,'OOF_WELLS'] = new_df.at[i,'OOF_WELLS_y']
				new_df.at[i,'ANALYSIS_TIME'] = now.strftime('%Y-%m-%d %H:%M:%S')

			else:
				new_df.at[i,'SCAN_TIME'] = new_df.at[i,'SCAN_TIME_x']
				new_df.at[i,'IS_OOF'] = new_df.at[i,'IS_OOF_x']
				new_df.at[i,'IF_COUNT'] = new_df.at[i,'IF_COUNT_x']
				new_df.at[i,'OOF_COUNT'] = new_df.at[i,'OOF_COUNT_x']
				new_df.at[i,'EMPTY_COUNT'] = new_df.at[i,'EMPTY_COUNT_x']
				new_df.at[i,'OOF_WELLS'] = new_df.at[i,'OOF_WELLS_x']
				new_df.at[i,'ANALYSIS_TIME'] = now.strftime('%Y-%m-%d %H:%M:%S')

		drop_cols = [c for c in new_df.columns if c.endswith('_x') or c.endswith('_y')]
		new_df.drop(columns=drop_cols,inplace=True)			
	new_df = new_df.sort_values(by=['RUN','PLATE'])
	new_df.to_csv(CSV_OUTPUT,index=False)



