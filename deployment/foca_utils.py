'''
Miscellaneous utilities for visualizing results and saving images
'''
import os
import re
import sys
from pathlib import Path
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).parents[1]))


# import pyodbc
# from slack_sdk import WebClient

# from input.credentials.keys import (DATABASE, DBPASSWORD, DBUSERNAME, SERVER,
#                                     SLACK_TOKEN) ## Uncomment if you've set up for the Slack features



def visualize_focal_classes(focus_dict,plate_name,save_dir,dpi=80):
	'''
	Creates plate layout plot as a matrix where:
	- Green: In-Focus
	- 	Red: Out-of-Focus
	-  Grey: Empty
	'''
	wells = list(focus_dict.keys())
	rows = []
	cols = []
	for well in wells:
		row,col = well[0].upper(),well[1:].zfill(2)
		rows.append(row)
		cols.append(col)
	rows = list(set(rows))
	cols = list(set(cols))
	matrix = np.zeros((len(rows),len(cols)))

	for well in wells:
		row,col = well[0].upper(),well[1:]
		matrix[ord(row)-65,int(col)-1] = focus_dict[well]

	colors=matplotlib.colors.ListedColormap(['green','red','gray'])
	plt.figure(figsize=(len(cols)/3,len(rows)/3))
	plt.imshow(matrix,cmap=colors,vmin=0,vmax=2)
	plt.xticks(np.arange(0.5,len(cols)+0.5,1))
	plt.yticks(np.arange(0.5,len(rows)+0.5,1))
	plt.gca().get_yaxis().set_ticklabels([])
	plt.gca().get_xaxis().set_ticklabels([])
	plt.grid()
	plt.title(plate_name)
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir,plate_name+'.png'),dpi=dpi,bbox_inches='tight',transparent=True)
	plt.close()


def save_image(dir,fname,save_name,save_dir,full_res=True):
	new_fname = "_".join([dir.split('/')[-1],save_name])
	if full_res:
		new_fname = re.sub(r"([0-9]+)um","1um",new_fname)
		new_fname = re.sub(r"TIFF([0-9]+)","TIFF01",new_fname)
	img = cv.imread(os.path.join(dir,fname),cv.IMREAD_UNCHANGED)
	cv.imwrite(os.path.join(save_dir,new_fname),img)