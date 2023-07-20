import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import pytest
import tensorflow as tf

from input.config import *


@pytest.mark.paths
def test_paths():
    print('Testing paths in input/config.py')
    assert os.path.exists(MODEL_WEIGHTS), "Check MODEL_WEIGHTS path in input/config.py"
    
    assert os.path.exists(IMAGE_DIR), "Check IMAGE_DIR path in input/config.py"
    if SORT_IMAGES == True:
        for i in range(len(FLAGGED_DIRS)):
            assert os.path.exists(FLAGGED_DIRS[i]), "Check that %s is a valid path"%FLAGGED_DIRS

    assert os.path.exists(FOCUS_VISUALIZATION_DIR), "Check FOCUS_VISUALIZATION_DIR path in input/config.py"

def test_patch_selection_method():
    assert PATCH_SELECTION_METHOD in ['mid_std_dev','max_std_dev','random'], "Invalid patch selection method: please choose from ['mid_std_dev','max_std_dev','random']"

def test_normalization_range():
    assert NORM_RANGE[0] < NORM_RANGE[1], "Normalization range must have positive size"
    assert 0 <= NORM_RANGE[0] <= 255, "Normalization range must be bounded (inclusively) in [0,255]"
    assert 0 <= NORM_RANGE[1] <= 255, "Normalization range must be bounded (inclusively) in [0,255]"

def test_sort_images_flag():
    assert type(SORT_IMAGES) is bool, "Choose True or False for SORT_IMAGES in input/config.py"

def test_model_weights():
    try:
        tf.keras.models.load_model(MODEL_WEIGHTS,compile=False)
        assert True
    except Exception:
        assert False, "MODEL_WEIGHTS cannot be loaded by Tensorflow"
