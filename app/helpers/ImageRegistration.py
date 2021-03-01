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
from app.controllers import Helper

class ImageRegistration():

    image_regist_result = ""

    def get_homography(self,img1, img2, algo, is_clahe):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

            # Initialize SIFT 
        if algo == 'sift':
            sift = cv2.xfeatures2d.SIFT_create()
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        if is_clahe == 1:
            img1 = clahe.apply(img1)
            img2 = clahe.apply(img2)

        # Extract keypoints and descriptors
        k1, d1 = sift.detectAndCompute(img1, None)
        k2, d2 = sift.detectAndCompute(img2, None)

        # Bruteforce matcher on the descriptors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(d1,d2, k=2)

        # Make sure that the matches are good
        verify_ratio = 0.8 # Source: stackoverflow
        verified_matches = []
        for m1,m2 in matches:
            # Add to array only if it's a good match
            if m1.distance < 0.8 * m2.distance:
                verified_matches.append(m1)

        # Mimnum number of matches
        min_matches = 8
        if len(verified_matches) > min_matches:
            
            # Array to store matching points
            img1_pts = []
            img2_pts = []

            # Add matching points to array
            for match in verified_matches:
                img1_pts.append(k1[match.queryIdx].pt)
                img2_pts.append(k2[match.trainIdx].pt)
            img1_pts = np.float32(img1_pts).reshape(-1,1,2)
            img2_pts = np.float32(img2_pts).reshape(-1,1,2)
            
            # Compute homography matrix
            M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
            return M
        else:
            return "fail"

    def get_stitched_image(self,img1, img2, M):
        # Get width and height of input images	
        w1,h1 = img1.shape[:2]
        w2,h2 = img2.shape[:2]

        # Get the canvas dimesions
        img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
        img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)


        # Get relative perspective of second image
        img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

        # Resulting dimensions
        result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

        # Getting images together
        # Calculate dimensions of match points
        [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

        # Create output array after affine transformation 
        transform_dist = [-x_min,-y_min]
        transform_array = np.array([[1, 0, transform_dist[0]], 
                        [0, 1, transform_dist[1]], 
                        [0,0,1]]) 

        # Warp images to get the resulting image
        result_img = cv2.warpPerspective(img2, transform_array.dot(M), 
                        (x_max-x_min, y_max-y_min))
        result_img[transform_dist[1]:w1+transform_dist[1], 
                transform_dist[0]:h1+transform_dist[0]] = img1
        theta = - math.atan2(M[0,1], M[0,0]) * 180 / math.pi
        '''
            0,0 0,1 0,2(tX)
            1,0 1,1 1,2(tY)
            2,0 2,1 2,2
        '''
        scale_x = M[0,0]
        scale_y = M[1,1]
        # Return the result
        return result_img, theta ,M[0,2], M[1,2], scale_x, scale_y

    def rmsdiff(self,im1, im2):
    # """Calculates the root mean square error (RSME) between two images"""
        return math.sqrt(mse(img_as_float(im1), img_as_float(im2)))

    def registration(self,reference_path, target_path, algo='sift', is_clahe=0):
                # Get input set of images
            img1 = cv2.imread(reference_path)
            img2 = cv2.imread(target_path)
            # img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)

            # Use SIFT to find keypoints and return homography matrix
            M =  self.get_homography(img1, img2, algo, is_clahe)
            if M == "fail":
                cv2.imwrite(os.path.abspath(os.curdir +"/uploads/result.png"),img1)

                with open(os.path.abspath(os.curdir +"/uploads/result.png"), "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read())
                    self.image_regist_result = b64_string.decode('utf-8')
                return 0,0,0,self.image_regist_result
            # Stitch the images together using homography matrix
            result_image, theta, tx, ty, scale_x, scale_y = self.get_stitched_image(img2, img1, M)
            rmse = self.rmsdiff(img1, cv2.resize(result_image, (img1.shape[1], img1.shape[0])))
            cv2.imwrite(os.path.abspath(os.curdir +"/uploads/result.png"),result_image)

            with open(os.path.abspath(os.curdir +"/uploads/result.png"), "rb") as img_file:
                b64_string = base64.b64encode(img_file.read())
                self.image_regist_result = b64_string.decode('utf-8')
            
            print(f"RMSE: {rmse}")
            print(f"tx: {tx}, ty: {ty}, theta: {theta}")
            print(f"theta: {theta}")

            return tx,ty,theta,self.image_regist_result