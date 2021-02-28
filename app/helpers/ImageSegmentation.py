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
from app.helpers.ImageProcessing import ImageProcessing 


class ImageSegmentation(ImageProcessing):

    def d2Kmeans(self,img, k):
        return KMeans(n_jobs=-1, 
                    random_state=1, 
                    n_clusters = k, 
                    init='k-means++'
        ).fit(img.reshape((-1,1))).labels_.reshape(img.shape)

    def binary(self,image):
        return image > threshold_otsu(image)

    def mean_filter(self,image, raio_disk):
        return rank.mean_percentile(image, selem = disk(raio_disk))

    def select_cluster_index(self,clusters):
        minx = clusters[0].mean()
        index = 0
        for i in clusters:
            if i.mean() < minx:
                minx = i.mean()
                index += 1
        return index

    def wathershed(self,image_path):
        im = ImageProcessing()
        watershed_image = ""
    
        im.image = cv2.imread(image_path)
        gray = cv2.cvtColor(im.image,cv2.COLOR_BGR2GRAY)
        image_binary = np.zeros(gray.shape, dtype=np.uint8)
        # ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 719, 20)
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 3)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers1 = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers1 + 20

        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        markers = cv2.watershed(im.image,markers)

        for i, label in enumerate(np.unique(markers)):
            if label == 0:
                continue        
            if i < 3:
                continue
            # Create a mask
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[markers == label] = 255

            # Find contours and determine contour area
            # mask = cv2.bitwise_not(mask)
            cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            # total_area += area
            cv2.drawContours(image_binary, [c], -1, (255,255,255), -1) #buat mask
            cv2.drawContours(im.image, [c], -1, (255,255, 255), 2) #buat gambar asli
            cv2.imwrite(os.path.abspath(os.curdir + "/uploads/result_watershed.jpg"),im.image)
            with open(os.path.abspath(os.curdir + "/uploads/result_watershed.jpg"), "rb") as img_file:
                b64_string = base64.b64encode(img_file.read())
                im.resultImage = b64_string.decode('utf-8')
    
        return im.resultImage

    def kmeans_segmentation(self,image_path,clusters_count):
        clusters_amount =  clusters_count
        kmeans_image = ""
        im = ImageProcessing()
        with open(image_path, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read())
            upload_image = b64_string.decode('utf-8')

        
            #'/content/drive/MyDrive/Dataset Asli/40X/Benign/adenosis/SOB_B_A-14-22549AB-40-001.png'
            im.image = cv2.imread(image_path)
            result_gray = self.d2Kmeans(rgb2grey(im.image), clusters_amount)
            clusters = [result_gray == i for i in range(clusters_amount)]
            cluster_index = self.select_cluster_index(clusters)
            
            selected_cluster = clusters[cluster_index]
            image_mean_filter = self.mean_filter(selected_cluster, 20)
            test_binary = self.binary(image_mean_filter)
            test_binary = test_binary.astype('uint8')
            contours, hierarchy = cv2.findContours(test_binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2:]
            boundary = cv2.drawContours(im.image, contours, -1, (255, 255, 255), 2)
            cv2.imwrite(os.path.abspath(os.curdir + "/uploads/result_kmeans.jpg"),im.image)
            with open(os.path.abspath(os.curdir + "/uploads/result_kmeans.jpg"), "rb") as img_file:
                b64_string = base64.b64encode(img_file.read())
                im.resultImage = b64_string.decode('utf-8')
            
            return im.resultImage