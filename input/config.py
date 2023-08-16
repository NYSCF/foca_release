'''
GLOBAL PARAMETERS
'''

# Dictionary for converting micrometer per pixel resolutions to equivalent pixel dimensions
SCALE_DICT = dict(zip(['10um','9um','8um','7um','6um','5um','4um','3um','2um','1um'],
                      [789,878,994,1111,1296,1562,1959,2622,3947,7520]))

# 8-bit grayscale range to normalize each well image in (min_value,max_value)
NORM_RANGE = (6,255)

'''
DEPLOYMENT PARAMETERS
'''

# Path to model weights
MODEL_WEIGHTS = 'input/pretrained_weights/final_config'

# Directory with plate subdirectories of images to analyze

IMAGE_DIR = 'input/deployment_data/'

# Set to True to sort images into folders to gather new data
SORT_IMAGES = False

# Paths of directories to sort images into if SORT_IMAGES is True
FLAGGED_DIRS = ['output/Saved_images_from_model/Flagged_IF',
                'output/Saved_images_from_model/Flagged_OOF']

# Path to save plate focus visualizations
FOCUS_VISUALIZATION_DIR = 'output/plate_viz'

# Slack User ID for posting slack notifications
# Insert your Slack user ID here if you want to use the Slack features
# SLACK_USER_ID = ''

# Desired path to output CSV, will be created if it does not exist
CSV_OUTPUT = 'output/summary.csv'

# How to select patches from a well based on standard deviation
PATCH_SELECTION_METHOD = 'mid_std_dev' # Recommended
# PATCH_SELECTION_METHOD = 'max_std_dev'
# PATCH_SELECTION_METHOD = 'random'

# Resolution of Data
DATA_RESOLUTION = '6um'

'''
TRAINING PARAMETERS
'''

# Path to directory with class subdirectories of well images
DATA_DIR = " " 

# Subdirectories of DATA_DIR for each focus class. First in list will be in-focus, second out-of-focus
# If you provide a list of multiple subdirectories for the same class, samples will be taken equally across them
DATA_CLASS_DIRS = ['InFocus','OutOfFocus']
# DATA_CLASS_DIRS = ['InFocus',['OutOfFocus1','OutOfFocus2']]

# Path to directory where CSV datasets are to be saved and read from
DATASET_CSV_DIR ="assets/datasets/"

# Path to directory with FFC imgs:
FFC_DIR = "assets/FFC/"

# Path to directory where model weights will be saved and loaded from
WEIGHT_DIR = "input/model_weights"

# Path to directory where experimental results will be saved and read from
EXPERIMENT_DIR = "assets/experiment_results"

# Path to directory with YAML config files for dataloaders and model parameters
CONFIG_DIR = "input/" 

# Path to directory where plots will be saved
PLOT_DIR = "assets/model_plots"

# Maximum number of wells to include in each class for training set
MAX_CLASS_SIZE = 200