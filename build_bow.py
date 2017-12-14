from params import get_params
import sys

# We need to add the source code path to the python path if we want to call modules such as 'utils'
params = get_params()
sys.path.insert(0,params['src'])

import os, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import pickle
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# build bow
def bow(assignments,km):

    # Initialize empty descriptor of the same length as the number of clusters
    descriptor = np.zeros(np.shape(km.cluster_centers_)[0])

    # Build vector of repetitions
    for a in assignments:

        descriptor[a] += 1
        
       
    # L2 normalize
    descriptor = normalize(descriptor)
   
    
    
    
    Path = mpath.Path
    
    fig, ax = plt.subplots()
    pp1 = mpatches.PathPatch(
        Path([(0, 0), (1, 0), (1, 1), (0, 0)],
            [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
        fc="none", transform=ax.transData)
    
    ax.add_patch(pp1)
    ax.plot(descriptor)
    ax.set_title('The red point should be on the path')
    
    plt.show()
        

    return descriptor

