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

# get local features
def image_local_features(im,detector,extractor):

    '''
    Extract local features for given image
    '''

    positions = detector.detect(im,None)
    positions, descriptors = extractor.compute(im,positions)

    return descriptors

