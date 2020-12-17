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

train_datas, test_datas = [], []
clahe_image= ""

class Helper():
  def init_dataset(data_list, dirpath):
    for folder, subfolder, file in os.walk(dirpath):
      dirname = folder.split(os.path.sep)[-1]
      main_class = folder.split(os.path.sep)[-3]
      for folder2, subfolder2, file2 in os.walk(folder):
        if dirname == "train" or dirname == "validation":
          continue
        data_list.append({
            "main_class": main_class,
            "class" : dirname,
            "path" : folder2,
            "filenames" : file2
        })
    return data_list
        
  def delete_class(datas, label):
    for data in datas:
      if data['class'] == label:
        datas.remove(data)
    return datab

  train_datas = init_dataset(train_datas, os.path.abspath(os.curdir +"/dataset/benign/train"))
  train_datas = init_dataset(train_datas,  os.path.abspath(os.curdir +"/dataset/malignant/train"))
  test_datas = init_dataset(test_datas, os.path.abspath(os.curdir +"/dataset/benign/validation"))
  test_datas = init_dataset(test_datas, os.path.abspath(os.curdir +"/dataset/malignant/validation"))

  count_train = 0
  for data in train_datas:
    count_train = count_train + len(data['filenames'])
    print(data)
  print("---")
  count_test = 0
  for data in test_datas:
    count_test = count_test + len(data['filenames'])
    print(data)

  print(count_train)
  print(count_test)

  x_train, y_train = [], []
  y_train_benign = []
  y_train_malignant = []
  y_test_benign = []
  y_test_malignant = []
  for data in train_datas:
    for i, filename in enumerate(data['filenames'][:20]):
      # if i == 50:
      #   break
      x_train.append(data['path'] + '/' + filename)
      y_train.append(data['class'])
      if data['main_class'] == 'Benign':
        y_train_benign.append(data['class'])
      else:
        y_train_malignant.append(data['class'])
  x_test, y_test = [], []
  for data in test_datas:
    for i, filename in enumerate(data['filenames'][:20]):
      # if i == 50:
      #   break
      x_test.append(data['path'] + '/' + filename)
      y_test.append(data['class'])
      if data['main_class'] == 'Benign':
        y_test_benign.append(data['class'])
      else:
        y_test_malignant.append(data['class'])




  def load_model(json_path, weight_path):
    json_file = open(json_path, 'r') #.json
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weight_path)
    print("Loaded model from disk")
    return loaded_model

  def load_pickle(path):
    infile = open(path,'rb')
    loaded_pickle = pickle.load(infile)
    infile.close()
    return loaded_pickle

  model = load_model(os.path.abspath(os.curdir + "/breakhist_biner_io/earlystopping/model-all-classification.json"),
                      os.path.abspath(os.curdir + "/breakhist_biner_io/earlystopping/model-all-classfication-weights.h5"))
  scale = load_pickle(os.path.abspath(os.curdir + "/breakhist_biner_io/earlystopping/scale"))
  kmeans = load_pickle(os.path.abspath(os.curdir + "/breakhist_biner_io/earlystopping/kmeans"))


  benign_model = load_model(os.path.abspath(os.curdir + "/breakhist_4_kelas_alldata_benign_stain/model_breakhist_4_subclass_benign.json"),
                      os.path.abspath(os.curdir + "/breakhist_4_kelas_alldata_benign_stain/model_breakhist_4_subclass_benign.h5"))
  benign_scale = load_pickle(os.path.abspath(os.curdir + "/breakhist_4_kelas_alldata_benign_stain/scale"))
  benign_kmeans = load_pickle(os.path.abspath(os.curdir + "/breakhist_4_kelas_alldata_benign_stain/kmeans"))

  malignant_model = load_model(os.path.abspath(os.curdir + "/breakhist_4_kelas_alldata_malignant_stain/model_breakhist_4class_malignant_farrel.json"),
                      os.path.abspath(os.curdir + "/breakhist_4_kelas_alldata_malignant_stain/model_breakhist_4class_malignant_farrel.h5"))
  malignant_scale = load_pickle(os.path.abspath(os.curdir + "/breakhist_4_kelas_alldata_malignant_stain/scale"))
  malignant_kmeans = load_pickle(os.path.abspath(os.curdir + "/breakhist_4_kelas_alldata_malignant_stain/kmeans"))

  num_words = 800
  sift = cv2.xfeatures2d.SIFT_create()
      
  adenosis = 0
  fibroadenoma = 0
  phyllodes_tumor = 0
  tubular_adenoma = 0

  ductal_carcinoma = 0
  lobular_carcinoma = 0
  mucinous_carcinoma = 0
  papillary_carcinoma = 0
  b_features_repository = []
  m_features_repository = []
  b_repository = []
  m_repository = []
  b_descriptor_list, m_descriptor_list = [], []
  b_labels, m_labels = [], []
  m_count, b_count = 0, 0


  b_features_repository = load_pickle(os.path.abspath(os.curdir + "/pickle/b_features_repository.pickle"))
  b_repository = load_pickle(os.path.abspath(os.curdir + "/pickle/b_repository.pickle"))
  b_descriptor_list = load_pickle(os.path.abspath(os.curdir + "/pickle/b_descriptor_list.pickle"))
  b_labels = load_pickle(os.path.abspath(os.curdir + "/pickle/b_labels.pickle"))

  m_features_repository = load_pickle(os.path.abspath(os.curdir + "/pickle/m_features_repository.pickle"))
  m_repository = load_pickle(os.path.abspath(os.curdir + "/pickle/m_repository.pickle"))
  m_descriptor_list = load_pickle(os.path.abspath(os.curdir + "/pickle/m_descriptor_list.pickle"))
  m_labels = load_pickle(os.path.abspath(os.curdir + "/pickle/m_labels.pickle"))

  print("sudah 10 semuanya")
  b_im_features = load_pickle(os.path.abspath(os.curdir + "/pickle/b_im_features.pickle"))
  m_im_features = load_pickle(os.path.abspath(os.curdir + "/pickle/m_im_features.pickle"))


  from sklearn.neighbors import KNeighborsClassifier

  from keras.utils import to_categorical
  from sklearn.preprocessing import LabelEncoder, StandardScaler

  labelencoder = LabelEncoder()
  y_benign = labelencoder.fit_transform(b_labels)
  y_benign = np.asarray(y_benign)
  y_benign = to_categorical(y_benign)
  y_malignant = labelencoder.fit_transform(m_labels)
  y_malignant = np.asarray(y_malignant)
  y_malignant = to_categorical(y_malignant)
  b_knn = KNeighborsClassifier(n_neighbors = 4) #define K=1
  print(np.array(b_im_features).shape)
  b_knn.fit(b_im_features, y_benign)

  m_knn = KNeighborsClassifier(n_neighbors = 4) #define K=1
  m_knn.fit(m_im_features, y_malignant)

  import pickle
