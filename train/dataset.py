import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parents[1]))
import itertools
import random
from datetime import datetime

import cv2 as cv
from dataloader import center_crop
from sklearn.model_selection import KFold, train_test_split

from input.config import DATASET_CSV_DIR, SCALE_DICT


def calculate_global_intensity_range(csv_names=('train.csv','test.csv'),scale='1um'):
    '''
    Calculate a global intensity range over a dataset for normalizing images

    Args:
        csv_names: tuple of strings
            CSV file names with training and test set information
    
    Returns:
        intensity_range: tuple
            tuple with min and max values for normalization (min, max)
    '''

    train_df = pd.read_csv(os.path.join(DATASET_CSV_DIR, csv_names[0]))
    test_df = pd.read_csv(os.path.join(DATASET_CSV_DIR,'test.csv'))

    dataset_df = pd.concat([train_df,test_df],ignore_index=True)
    global_min,global_max = 10000000,0
    for i in range(len(dataset_df)):
        well_info = dataset_df.iloc[i]
        ROI = np.round(SCALE_DICT[scale]*.4).astype(int)
        img = cv.imread(os.path.join(well_info['Location'],well_info['Filename']),cv.IMREAD_GRAYSCALE).astype(np.float32)

        img = center_crop(img,ROI=ROI)

        sample_min,sample_max = np.min(img),np.max(img)

        if sample_min < global_min:
            global_min = sample_min
            global_min_file = os.path.join(well_info['Location'],well_info['Filename'])
        if sample_max > global_max:
            global_max = sample_max
            global_max_file = os.path.join(well_info['Location'],well_info['Filename'])

    
    return (global_min,global_max),(global_min_file,global_max_file)


def create_datasets(data_dir,class_dirs,folds=5,split=0.3,max_class_size=50,csv_prefixes=('train','test'),generic=False):
    '''
    Automatically generates and saves CSVs for with training and testing sets where plate variety is maximized

    Args:
        data_dir: string
            parent directory where class directories can be found
        
        class_dirs: list of strings
            list of class directories where images in each class can be found
            if an element of the list is a list, this will include images from both directories
        
        split: float
            fraction of images to use in testing set
        
        max_class_size: int
            maximum number of samples used from any class

        csv_names: list of strings
            names of files to save training and test sets to
            defaults are "train.csv" and "test.csv"

        generic: bool
            if true, assigns all samples the same plate
            used if data format is different from standard
    Returns:
        train_df: pd.DataFrame
            DataFrame with training data
        
        test_df: pd.DataFrame
            DataFrame with testing data
    '''
    train_df = pd.DataFrame(columns=['Location','Filename','Plate','Class'])
    test_df = pd.DataFrame(columns=['Location','Filename','Plate','Class'])
    total_df = pd.DataFrame(columns=['Location','Filename','Plate','Class'])

    # get all files in the class and subclass directories 
    for i in range(len(class_dirs)):
        class_df = pd.DataFrame(columns=total_df.columns)
        if type(class_dirs[i]) is list:
            files = []
            for j in range(len(class_dirs[i])):
                subclass_dir = class_dirs[i][j]
                class_path = os.path.join(data_dir,subclass_dir)
                subclass_files = [os.path.join(class_path,f) for f in os.listdir(class_path) if f.endswith(('.jpg','.png','.tiff'))]
                files = files + subclass_files
        else:
            class_path = os.path.join(data_dir,class_dirs[i])
            files = [os.path.join(class_path,f) for f in os.listdir(class_path) if f.endswith(('.jpg','.png','.tiff'))]
        files = sorted(files)
        plates = []
        locations = []
        filenames = []

        # extract location, plate, and filename from absolute paths
        for j in range(len(files)):
            file_list = files[j].split('/')
            filename = file_list[-1]
            location = "/".join(file_list[:-1])
            filenames.append(filename)
            locations.append(location)
            if generic:
                plate = 'generic'
            else:
                plate = "_".join(filename.split('_')[:4]).split('Well')[0]
            plates.append(plate)
        
        class_df['Location'] = locations
        class_df['Filename'] = filenames
        class_df['Plate'] = plates
        class_df['Class'] = i
        
        # 
        if len(class_df) > max_class_size:
            class_df['Key']=class_df.groupby(['Location','Plate']).cumcount()
            class_df['Frequency'] = class_df.groupby(['Location','Plate'])['Plate'].transform('count')
            
            class_df = class_df.sort_values(['Key','Frequency'],ascending=[True,True]).drop(['Key','Frequency'],axis=1).iloc[:max_class_size]
            
        total_df = pd.concat([total_df,class_df],ignore_index=True)
    
    if folds==1:
        total_df.to_csv(os.path.join(DATASET_CSV_DIR,csv_prefixes[0]+'%dwell.csv'%max_class_size))
        return csv_prefixes[0]+'%dwell.csv'%max_class_size,None
    train_fold_dfs = []
    test_fold_dfs = []
    for i in range(folds):
        train_fold_dfs.append(pd.DataFrame(columns=total_df.columns))
        test_fold_dfs.append(pd.DataFrame(columns=total_df.columns))

    kf = KFold(n_splits=folds,shuffle=False)
    for i in range(len(class_dirs)):
        class_df = total_df[total_df['Class']==i]
        kf.get_n_splits(class_df)

        unique_locations = class_df['Location'].unique()
        for j in range(len(unique_locations)):
            subclass_df = class_df[class_df['Location']==unique_locations[j]]

        
            for k, (train_index,test_index) in enumerate(kf.split(subclass_df)):
                # print('Train: ',train_index)
                # print('Test: ',test_index)
                train_df = subclass_df.iloc[train_index]
                test_df = subclass_df.iloc[test_index]
                train_fold_dfs[k] = pd.concat([train_fold_dfs[k],train_df],ignore_index=True)
                test_fold_dfs[k] = pd.concat([test_fold_dfs[k],test_df],ignore_index=True)
    
    for i in range(folds):
        train_fold_dfs[i].to_csv(os.path.join(DATASET_CSV_DIR,csv_prefixes[0]+'_f%d.csv'%i))
        test_fold_dfs[i].to_csv(os.path.join(DATASET_CSV_DIR,csv_prefixes[1]+'_f%d.csv'%i))
    


    return csv_prefixes[0]+'_f0.csv',csv_prefixes[1]+'_f0.csv'


    

