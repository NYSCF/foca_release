import pytest
import warnings
import os,sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from input.config import *
@pytest.mark.paths
def test_paths():
    print('Checking paths used in training')
    assert os.path.exists(DATA_DIR), "Check DATA_DIR path in input/config.py"
    
    for i in range(len(DATA_CLASS_DIRS)):
        if type(DATA_CLASS_DIRS[i]) is list:
            for j in range(len(DATA_CLASS_DIRS[i])):
                class_dir = os.path.join(DATA_DIR,DATA_CLASS_DIRS[i][j])
                os.path.exists(class_dir), f"{DATA_CLASS_DIRS[i][j]} is not in {DATA_DIR}; check DATA_CLASS_DIRS in input/config.py"
        else:
            class_dir = os.path.join(DATA_DIR,DATA_CLASS_DIRS[i])
            assert os.path.exists(class_dir), f"{DATA_CLASS_DIRS[i]} is not in {DATA_DIR}; check DATA_CLASS_DIRS in input/config.py"
    
    assert os.path.exists(DATASET_CSV_DIR), "Check DATASET_CSV path in input/config.py"
    
    assert os.path.exists(WEIGHT_DIR), "Check WEIGHT_DIR path in input/config.py"

    assert os.path.exists(EXPERIMENT_DIR), "Check EXPERIMENT_DIR path in input/config.py"

    assert os.path.exists(CONFIG_DIR), "Check CONFIG_DIR path in input/config.py"

    assert os.path.exists(PLOT_DIR), "Check PLOT_DIR path in input/config.py"

def test_normalization_range():
    assert NORM_RANGE[0] < NORM_RANGE[1], "Normalization range must have positive size"
    assert 0 <= NORM_RANGE[0] <= 255, "Normalization range must be bounded (inclusively) in [0,255]"
    assert 0 <= NORM_RANGE[1] <= 255, "Normalization range must be bounded (inclusively) in [0,255]"

def test_max_class_size():
    assert MAX_CLASS_SIZE > 0, "MAX_CLASS_SIZE must be greater than 0"
    if MAX_CLASS_SIZE < 30:
        warnings.warn(UserWarning('Using a small MAX_CLASS_SIZE will lead to poorer performance, consider setting MAX_CLASS_SIZE to at least 30'))

def test_scale_dict():
    if SCALE_DICT != dict(zip(['10um','9um','8um','7um','6um','5um','4um','3um','2um','1um'],[789,878,994,1111,1296,1562,1959,2622,3947,7520])):
        warnings.warn(UserWarning('Scale dictionary has been changed, ignore this if you changed it intentionally; otherwise, consult the github repo for the original values'))
