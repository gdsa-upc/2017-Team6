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

# train codebook
def train_codebook(params,X):

    # Init kmeans instance
    km = MiniBatchKMeans(params['descriptor_size'])

    # Training the model with our descriptors
    km.fit(X)

    # Save to disk
    pickle.dump(km,open(os.path.join(params['root'],params['root_save'],
                                     params['codebooks_dir'],'codebook_'
                                     + str(params['descriptor_size']) + "_"
                                     + params['descriptor_type']
                                     + "_" + params['keypoint_type'] + '.cb'),'wb'))

    return km

