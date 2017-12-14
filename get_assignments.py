from params import get_params
import sys

# We need to add the source code path to the python path if we want to call modules such as 'utils'
params = get_params()
sys.path.insert(0,params['src'])

import os, time
import numpy as np
import pickle
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# get features
def get_assignments(km,descriptors):

    assignments = km.predict(descriptors)

    return assignments

