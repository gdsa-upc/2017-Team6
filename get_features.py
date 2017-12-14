from params import get_params
import sys

# We need to add the source code path to the python path if we want to call modules such as 'utils'
params = get_params()
sys.path.insert(0,params['src'])

from train_codebook import train_codebook
from get_assignments import get_assignments
from build_bow import bow
from get_local_features import image_local_features

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
def get_features(params,pca=None,scaler=None):
    
    # Read image names
    with open(os.path.join(params['root'],params['root_save'],params['image_lists'],params['split'] + '.txt'),'r') as f:
        image_list = f.readlines()

    # Initialize keypoint detector and feature extractor
    detector, extractor = init_detect_extract(params)

    # Initialize feature dictionary
    features = {}

    # Get trained codebook
    km = pickle.load(open(os.path.join(params['root'],params['root_save'],
                                     params['codebooks_dir'],'codebook_'
                                     + str(params['descriptor_size']) + "_"
                                     + params['descriptor_type']
                                     + "_" + params['keypoint_type'] + '.cb'),'rb'))

    for image_name in image_list:

        # Read image
        im = cv2.imread(os.path.join(params['root'],params['database'],params['split'],'images',image_name.rstrip()))

        # Resize image
        im = resize_image(params,im)

        # Extract local features
        feats = image_local_features(im,detector,extractor)

        if feats is not None:
            
            if params['normalize_feats']:
                feats = normalize(feats)
            
            # If we scaled training features
            if scaler is not None:
                scaler.transform(feats)
            
            # Whiten if needed
            if pca is not None:
                
                pca.transform(feats)

            # Compute assignemnts
            assignments = get_assignments(km,feats)

            # Generate bow vector
            feats = bow(assignments,km)
        else:
            # Empty features
            feats = np.zeros(params['descriptor_size'])

        # Add entry to dictionary
        features[image_name] = feats


    # Save dictionary to disk with unique name
    save_file = os.path.join(params['root'],params['root_save'],params['feats_dir'],
                             params['split'] + "_" + str(params['descriptor_size']) + "_"
                             + params['descriptor_type'] + "_" + params['keypoint_type'] + '.p')

    pickle.dump(features,open(save_file,'wb'))

# Cambia tamany imatge
def resize_image(params,im):

    # Get image dimensions
    height, width = im.shape[:2]

    # If the image width is smaller than the proposed small dimension, keep the original size !
    resize_dim = min(params['max_size'],width)

    # We don't want to lose aspect ratio:
    dim = (resize_dim, height * resize_dim/width)

    # Resize and return new image
    return cv2.resize(im,dim)

# 
def init_detect_extract(params):

    '''
    Initialize detector and extractor from parameters
    '''
    if params['descriptor_type'] == 'RootSIFT':
        
        extractor = RootSIFT()
    else:
        
        extractor = cv2.DescriptorExtractor_create(params['descriptor_type'])
        
    detector = cv2.FeatureDetector_create(params['keypoint_type'])

    return detector, extractor

# 
def stack_features(params):

    '''
    Get local features for all training images together
    '''

    # Init detector and extractor
    detector, extractor = init_detect_extract(params)

    # Read image names
    with open(os.path.join(params['root'],params['root_save'],params['image_lists'],params['split'] + '.txt'),'r') as f:
        image_list = f.readlines()

    X = []
    for image_name in image_list:

        # Read image
        im = cv2.imread(os.path.join(params['root'],params['database'],params['split'],'images',image_name.rstrip()))

        # Resize image
        im = resize_image(params,im)

        feats = image_local_features(im,detector,extractor)
        # Stack all local descriptors together

        if feats is not None:
            if len(X) == 0:

                X = feats
            else:
                X = np.vstack((X,feats))
                
    if params['normalize_feats']:
        X = normalize(X)
    
    if params['whiten']:
        
        pca = PCA(whiten=True)
        pca.fit_transform(X)
        
    else:
        pca = None
    
    # Scale data to 0 mean and unit variance
    if params['scale']:
        
        scaler = StandardScaler()
        
        scaler.fit_transform(X)
    else:
        scaler = None
    
    return X, pca, scaler

if __name__ == "__main__":

    params = get_params()

    # Change to training set
    params['split'] = 'train'
    
    print "Apilant descriptors..."
    # Save features for training set
    t = time.time()
    X, pca, scaler = stack_features(params)
    print "Fet! Temps utilitzat:", time.time() - t
    print "Nombre de descriptors d'entrenament", np.shape(X)

    print "Entrenant codebook..."
    t = time.time()
    train_codebook(params,X)
    print "Fet! Temps utilitzat:", time.time() - t
    
    print "Emmagatzemant baul de descriptors per al set d'entrenament..."
    t = time.time()
    get_features(params, pca,scaler)
    print "Fet! Temps utilitzat:", time.time() - t

    params['split'] = 'val'
    
    print "Emmagatzemant baul de descriptors per al set de validacio..."
    t = time.time()
    get_features(params)
    print "Fet! Temps utilitzat", time.time() - t

