from app.controllers import Helper
import cv2
import numpy as np 
import pickle
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow.python.keras
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, Activation, MaxPooling1D, MaxPooling2D, GlobalAveragePooling1D,Reshape, Input
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.initializers import RandomNormal
import pickle
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from scipy.cluster.vq import *
from sklearn import preprocessing
import joblib
import random
import pickle
from numpy import dot
from numpy.linalg import norm
import base64
from flask import jsonify
from flask import request
import shutil
from flask_cors import CORS,cross_origin
from PIL import Image # No need for ImageChops
import math
from skimage import img_as_float
from skimage.metrics import mean_squared_error as mse
import staintools
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import imutils
import cv2
from glob import glob
from skimage.io import imread
from skimage.color import rgb2grey
from sklearn.feature_extraction import image
from sklearn.cluster import KMeans
from skimage.filters import rank, threshold_otsu
from skimage.morphology import closing, square, disk
from skimage import exposure as hist, data, img_as_float
from skimage.segmentation import chan_vese
from skimage.feature import canny
from skimage.color import rgb2gray
from scipy import ndimage as ndi 

class ImageEnhancement():
    
    valueBtn = ''
    resultImage = ''

    def claheImage(self,img_path):
        img = cv2.imread(img_path,0)  
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        cv2.imwrite(os.path.abspath(os.curdir +"/uploads/clahe.jpg"),img)
        with open(os.path.abspath(os.curdir +"/uploads/clahe.jpg"), "rb") as img_file:
            b64_string = base64.b64encode(img_file.read())
            resultImage = b64_string.decode('utf-8')
        return resultImage

    def stainImage(self,trg_path,ref_path):
        target = staintools.read_image(trg_path)
        to_transform = staintools.read_image(ref_path)  

        normalizer = staintools.StainNormalizer(method='vahadane')
        normalizer.fit(target)
        transformed = normalizer.transform(to_transform)
        img_transformed = cv2.resize(transformed,(300,300))
        cv2.imwrite(os.path.abspath(os.curdir +"/uploads/result_stain.jpg"),img_transformed)

        with open(os.path.abspath(os.curdir +"/uploads/result_stain.jpg"), "rb") as img_file:
            b64_string = base64.b64encode(img_file.read())
            img_result_stain_b64 = b64_string.decode('utf-8')

        return img_result_stain_b64