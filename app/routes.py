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
from app import app
import base64
from flask import jsonify
from flask import render_template

def getDescriptors(sift, img):
    kp, des = sift.detectAndCompute(img, None)
    return des

def readImage(img_path):
    # print(img_path)
    img = cv2.imread(img_path, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    return cv2.resize(img, (300,300))

def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor)) 

    return descriptors

def clusterDescriptors(descriptors, no_clusters):
    kmeans = MiniBatchKMeans(n_clusters = no_clusters).fit(descriptors)
    return kmeans

def extractFeatures(kmeans, descriptor_list, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
      try:
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1
      except:
        continue
    return im_features

def normalizeFeatures(scale, features):
    return scale.transform(features)

def plotHistogram(im_features, no_clusters):
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:,h], dtype=np.int32)) for h in range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()

def load_pickle(path):
  infile = open(path,'rb')
  loaded_pickle = pickle.load(infile)
  infile.close()
  return loaded_pickle

  
def load_model(json_path, weight_path):
  json_file = open(json_path, 'r') #.json
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  loaded_model.load_weights(weight_path)
  print("Loaded model from disk")
  return loaded_model

from tensorflow.python.keras.models import model_from_json
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from scipy.cluster.vq import *

sift = cv2.xfeatures2d.SIFT_create()

main_kmeans = load_pickle(os.path.abspath(os.curdir + "/!!model/!!model/main class model/main_class_kmeans"))
main_model = load_model(os.path.abspath(os.curdir + '/!!model/!!model/main class model/main_class_model.json'),
                     os.path.abspath(os.curdir + '/!!model/!!model/main class model/main_class_model.h5'))
main_scaler = load_pickle(os.path.abspath(os.curdir + '/!!model/!!model/main class model/main_class_scale'))

breakhist_kmeans = load_pickle(os.path.abspath(os.curdir + '/!!model/!!model/breakhist_binary_model/breakhist_binary_kmeans'))
breakhist_model = load_model(os.path.abspath(os.curdir + '/!!model/!!model/breakhist_binary_model/breakhist_binary.json'),
                     os.path.abspath(os.curdir + '/!!model/!!model/breakhist_binary_model/breakhist_binary.h5'))
breakhist_scaler = load_pickle(os.path.abspath(os.curdir + '/!!model/!!model/breakhist_binary_model/breakhist_binary_scale'))

breakhist_benign_kmeans = load_pickle(os.path.abspath(os.curdir + '/!!model/!!model/breakhist_benign_model/breakhist_benign_kmeans'))
breakhist_benign_model = load_model(os.path.abspath(os.curdir + '/!!model/!!model/breakhist_benign_model/breakist_benign_model.json'),
                     os.path.abspath(os.curdir + '/!!model/!!model/breakhist_benign_model/breakist_benign_model.h5'))
breakhist_benign_scaler = load_pickle(os.path.abspath(os.curdir + '/!!model/!!model/breakhist_benign_model/breakhist_benign_scale'))

breakhist_malignant_kmeans = load_pickle(os.path.abspath(os.curdir + '/!!model/!!model/breakhist_malignant/breakhist_malignant_kmeans'))
breakhist_malignant_model = load_model(os.path.abspath(os.curdir + '/!!model/!!model/breakhist_malignant/breakist_malignant_model.json'),
                     os.path.abspath(os.curdir + '/!!model/!!model/breakhist_malignant/breakist_malignant_model.h5'))
breakhist_malignant_scaler = load_pickle(os.path.abspath(os.curdir + '/!!model/!!model/breakhist_malignant/breakhist_malignant_scale'))

idrid_kmeans = load_pickle(os.path.abspath(os.curdir + '/!!model/!!model/idrid_model/kmeans_IDRID_2class'))
idrid_model = load_model(os.path.abspath(os.curdir + '/!!model/!!model/idrid_model/model_IDRID_2class.json'),
                     os.path.abspath(os.curdir + '/!!model/!!model/idrid_model/model_IDRID_2class.h5'))
idrid_scaler = load_pickle(os.path.abspath(os.curdir + '/!!model/!!model/idrid_model/scale'))

isic_kmeans = load_pickle(os.path.abspath(os.curdir + '/!!model/!!model/ISIC NW 10/kmeans_isic_binary_nw_10'))
isic_model = load_model(os.path.abspath(os.curdir + '/!!model/!!model/ISIC NW 10/model_isic_binary_nw_10.json'),
                     os.path.abspath(os.curdir + '/!!model/!!model/ISIC NW 10/model_isic_binary_nw_10.h5'))
isic_scaler = load_pickle(os.path.abspath(os.curdir + '/!!model/!!model/ISIC NW 10/isic_scale'))

mias_kmeans = load_pickle(os.path.abspath(os.curdir + '/!!model/!!model/mias_model/mias_kmeans'))
mias_model = load_model(os.path.abspath(os.curdir + '/!!model/!!model/mias_model/mias_model.json'),
                     os.path.abspath(os.curdir + '/!!model/!!model/mias_model/mias_model.h5'))
mias_scaler = load_pickle(os.path.abspath(os.curdir + '/!!model/!!model/mias_model/mias_scale'))


images_result = []
showing_labels = []
distance_result = []
class_result = ""
def mias_crop(input_data):
  img = cv2.imread(input_data, 0)

  methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                          'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
  try:
      pathTemplate = (os.path.abspath(os.curdir + '/app/template.png'))
      template = cv2.imread(pathTemplate,0)
  except IOError as e:
      print("({})".format(e))
  else:
      w, h = template.shape[::-1]

  method = eval(methods[5])

  try:
    res = cv2.matchTemplate(img, template, method)
  except:
    return input_data

  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
  if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
      top_left = min_loc
  else:
      top_left = max_loc
  bottom_right = (top_left[0] + w, top_left[1] + h)
  
  roi = img[top_left[1]:top_left[1]+h,0:bottom_right[1]+w]
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  roi = clahe.apply(roi)
  roi = cv2.cvtColor(roi,cv2.COLOR_GRAY2RGB)
  roi = cv2.resize(roi, (300, 300))
  cv2.imwrite('mias.png', roi)
  if len(roi) == 0:
    img = clahe.apply(img)
    return img
  return roi

def predict(kmeans, model, input_data, num_words, scaler, mias=-1):
  descriptor_list = []
  if mias == -1:
    img = readImage(input_data)
  else:
    img = input_data
  des = getDescriptors(sift, img)
  descriptor_list.append(des)
  im_features = extractFeatures(kmeans, descriptor_list, 1, num_words)
  im_features = scaler.transform(im_features)
  res = np.argmax(model.predict(im_features))
  return model.predict(im_features)

def predict_mias(input_data, kmeans, model, scaler):
  input_data = mias_crop(input_data)
  res = predict(kmeans, model, input_data, 300, scaler, 1)
  return res
  if res == 0:
    return "mias_benign"
  elif res == 1:
    return "mias_malignant"

def predict_main_class(input_data, kmeans, model, scaler):
  res = predict(kmeans, model, input_data, 100, scaler)
  return res
  
def predict_breakhist(input_data, kmeans, model, scaler):
  res = predict(kmeans, model, input_data, 320, scaler)
  return res

def predict_breakhist_benign(input_data, kmeans, model, scaler):
  res = predict(kmeans, model, input_data, 300, scaler)
  return res

def predict_breakhist_malignant(input_data, kmeans, model, scaler):
  res = predict(kmeans, model, input_data, 1024, scaler)
  return res

def predict_isic(input_data, kmeans, model, scaler):
  res = predict(kmeans, model, input_data, 10, scaler)
  return res
    

def predict_idrid(input_data, kmeans, model, scaler):
  res = predict(kmeans, model, input_data, 100, scaler)
  return res
    
def get_class(path):
  output = []
  main_class = predict_main_class(path,
                        main_kmeans, main_model, main_scaler
            )
  output.append(main_class)
  if np.argmax(main_class) == 3: #mias
    child_class = predict_mias(path,
                        mias_kmeans, mias_model, mias_scaler
            )
    output.append(child_class)

  elif np.argmax(main_class) == 1: #idrid
    child_class = predict_idrid(path,
                        idrid_kmeans, idrid_model, idrid_scaler
            )
    output.append(child_class)

  elif np.argmax(main_class) == 2: #isic
    child_class = predict_isic(path,
                            isic_kmeans, isic_model, isic_scaler
                )
    output.append(child_class)
  elif np.argmax(main_class) == 0: #breakhist
    breakhist_class = predict_breakhist(path,
                            breakhist_kmeans, breakhist_model, breakhist_scaler
                )
    output.append(breakhist_class)

    if np.argmax(breakhist_class) == 0:
      child_class = predict_breakhist_benign(path,
                              breakhist_benign_kmeans, breakhist_benign_model, breakhist_benign_scaler
                  )
      
      output.append(child_class)
    elif np.argmax(breakhist_class) == 1:
      child_class = predict_breakhist_malignant(path,
                              breakhist_malignant_kmeans, breakhist_malignant_model, breakhist_malignant_scaler
                  )
      output.append(child_class)
      
  return output
# from google.colab import files
# uploaded = files.upload()
# for fn in uploaded.keys():
#   print(predict_main_class(fn))

import random, datetime
def get_image_paths(path, count=10):
  random.seed(datetime.datetime.now())
  training_names = os.listdir(path)
  image_paths = []
  for training_name in training_names:
      image_path = os.path.join(path, training_name)
      image_paths += [image_path]
  if count == -1:
    return image_paths
  # return image_paths[:10]
  return random.sample(image_paths, count)

def train_bovw(path, classname, no_clusters=50):
  feature = []
  count=0
  image_paths = []
  sift = cv2.xfeatures2d.SIFT_create()
  descriptor_list = []
  label = []
  
  for img_path in path:
    print(img_path)
    img = readImage(img_path)
    des = getDescriptors(sift, img)
    if des is None:
      continue
    else:
      count = count + 1
      print(count)
      descriptor_list.append(des)
      label.append(classname)
      print(classname)

  descriptors = vstackDescriptors(descriptor_list)
  print("Descriptors vstacked.")
  kmeans = clusterDescriptors(descriptors, no_clusters)
  print("Descriptors clustered.")
  im_features = extractFeatures(kmeans, descriptor_list, count, no_clusters)
  print("Images features extracted.")
  scaler = StandardScaler().fit(im_features)        
  im_features = scaler.transform(im_features)
  print("Train images normalized.")
  return im_features, kmeans, scaler

def euclidean(a, b):
	return np.linalg.norm(a - b)
 
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
  return dot(a, b)/(norm(a)*norm(b))

def perform_search(queryFeatures, feats, maxResults=10):
	results = []
	for i in range(0, len(feats)):
		d = cosine_similarity(queryFeatures, feats[i])
		results.append((d, i))
	results = sorted(results, reverse=True)[:maxResults] # dari besar ke kecil (descending)
	count = 0
	return results

def get_query_features(kmeans, input_data, num_words, scaler):
  descriptor_list = []
  
  img = readImage(input_data)
  des = getDescriptors(sift, img)
  descriptor_list.append(des)
  im_features = extractFeatures(kmeans, descriptor_list, 1, num_words)
  im_features = scaler.transform(im_features)
  return im_features

  
def show_retrieved_images(query_path, image_paths, features, scaler, kmeans, numWords= 50):
  query_features = get_query_features(kmeans, query_path, numWords, scaler)
  results = perform_search(query_features, features)
  images = []
  path = []
  distances = []
  for (d, j) in results:
    image = cv2.imread(image_paths[j])
    images.append(image)
    distances.append(d)
    path.append(image_paths[j])
  print(f"Query:")
  print(f"Actual Class : {get_ground_truth(query_path)}")
  print(j)


  fig=plt.figure(figsize=(20, 8))
  columns = 5
  rows = 2
  for i in range(1, columns*rows +1):
    try:
      img = images[i-1]
      ax = fig.add_subplot(rows, columns, i)
      showing_label = "R" if get_ground_truth(query_path) == get_ground_truth(image_paths[i-1]) else "NR"
      # ax.set_title(f'{showing_label} - {distances[i-1][0]}')
      # plt.imshow(img)
      showing_labels.append(showing_label)
      distance_result.append(distances[i-1][0])
      images_result.append(image_paths[i-1])
      # cv2.imwrite(f"image{i}.png",img)
      print(image_paths[i-1])
      printf("fff")
      print(showing_labels[i])
    except:
      print("")
  # plt.show()

  #Breakhist Benign
breakhist_adenosis_path = os.path.abspath(os.curdir + '/dataset/BreakHist Split/Benign/train/adenosis')
breakhist_fibroadenoma_path = os.path.abspath(os.curdir + '/dataset/BreakHist Split/Benign/train/fibroadenoma')
breakhist_phyllodes_tumor_path = os.path.abspath(os.curdir + '/dataset/BreakHist Split/Benign/train/phyllodes_tumor')
breakhist_tubular_adenoma_path = os.path.abspath(os.curdir + '/dataset/BreakHist Split/Benign/train/tubular_adenoma')

breakhist_adenosis_path = get_image_paths(breakhist_adenosis_path)
breakhist_fibroadenoma_path = get_image_paths(breakhist_fibroadenoma_path)
breakhist_phyllodes_tumor_path = get_image_paths(breakhist_phyllodes_tumor_path)
breakhist_tubular_adenoma_path = get_image_paths(breakhist_tubular_adenoma_path)

adenosis_features, adenosis_kmeans, adenosis_scaler = train_bovw(breakhist_adenosis_path, 'adenosis', 300)
fibroadenoma_features, fibroadenoma_kmeans, fibroadenoma_scaler = train_bovw(breakhist_adenosis_path, 'fibroadenoma', 300)
phyllodes_tumor_features, phyllodes_tumor_kmeans, phyllodes_tumor_scaler = train_bovw(breakhist_adenosis_path, 'phyllodes_tumor', 300)
tubular_adenoma_features, tubular_adenoma_kmeans, tubular_adenoma_scaler = train_bovw(breakhist_adenosis_path, 'tubular_adenoma', 300)

#Breakhist Malignant
breakhist_ductal_carcinoma_path = os.path.abspath(os.curdir + '/dataset/BreakHist Split/Malignant/train/ductal_carcinoma')
breakhist_lobular_carcinoma_path = os.path.abspath(os.curdir + '/dataset/BreakHist Split/Malignant/train/lobular_carcinoma')
breakhist_mucinous_carcinoma_path = os.path.abspath(os.curdir + '/dataset/BreakHist Split/Malignant/train/mucinous_carcinoma')
breakhist_papillary_carcinoma_path = os.path.abspath(os.curdir + '/dataset/BreakHist Split/Malignant/train/papillary_carcinoma')

breakhist_ductal_carcinoma_path = get_image_paths(breakhist_ductal_carcinoma_path)
breakhist_lobular_carcinoma_path = get_image_paths(breakhist_lobular_carcinoma_path)
breakhist_mucinous_carcinoma_path = get_image_paths(breakhist_mucinous_carcinoma_path)
breakhist_papillary_carcinoma_path = get_image_paths(breakhist_papillary_carcinoma_path)


ductal_carcinoma_features, ductal_carcinoma_kmeans, ductal_carcinoma_scaler = train_bovw(breakhist_adenosis_path, 'ductal_carcinoma', 1024)
lobular_carcinoma_features, lobular_carcinoma_kmeans, lobular_carcinoma_scaler = train_bovw(breakhist_adenosis_path, 'lobular_carcinoma', 1024)
mucinous_carcinoma_features, mucinous_carcinoma_kmeans, mucinous_carcinoma_scaler = train_bovw(breakhist_adenosis_path, 'mucinous_carcinoma', 1024)
papillary_carcinoma_features, papillary_carcinoma_kmeans, papillary_carcinoma_scaler = train_bovw(breakhist_adenosis_path, 'papillary_carcinoma', 1024)

#IDRID Positive
idrid_positive_path = os.path.abspath(os.curdir + '/dataset/IDRiD Splitted/train/symptoms')

idrid_positive_path = get_image_paths(idrid_positive_path)

idrid_positive_features, idrid_positive_kmeans, idrid_positive_scaler = train_bovw(idrid_positive_path, 'symptoms', 100)

#IDRID Negative
idrid_negative_path = os.path.abspath(os.curdir + '/dataset/IDRiD Splitted/train/nosymptoms')

idrid_negative_path = get_image_paths(idrid_negative_path)

idrid_negative_features, idrid_negative_kmeans, idrid_negative_scaler = train_bovw(idrid_negative_path, 'nosymptoms', 100)

#ISIC Benign
isic_benign_path = os.path.abspath(os.curdir + '/dataset/ISIC-V2/train/benign')

isic_benign_path = get_image_paths(isic_benign_path)

isic_benign_features, isic_benign_kmeans, isic_benign_scaler = train_bovw(isic_benign_path, 'isic-benign', 10)
#ISIC Malignant
isic_malignant_path = os.path.abspath(os.curdir + '/dataset/ISIC-V2/train/malignant')

isic_malignant_path = get_image_paths(isic_malignant_path)

isic_malignant_features, isic_malignant_kmeans, isic_malignant_scaler = train_bovw(isic_malignant_path, 'isic-malignant', 10)

#MIAS Benign
mias_benign_path = os.path.abspath(os.curdir + '/dataset/Mias Split/train/mias-benign')

mias_benign_path = get_image_paths(mias_benign_path)

mias_benign_features, mias_benign_kmeans, mias_benign_scaler = train_bovw(mias_benign_path, 'mias-benign', 300)

#MIAS malignant
mias_malignant_path = os.path.abspath(os.curdir + '/dataset/Mias Split/train/mias-malignant')

mias_malignant_path = get_image_paths(mias_malignant_path)

mias_malignant_features, mias_malignant_kmeans, mias_malignant_scaler = train_bovw(mias_malignant_path, 'mias-malignant', 300)

main_class_label =  ['breakhist', 'idrid', 'isic', 'mias']
mias_label = ['mias_benign', 'mias_malignant']
breakhist_label = ['benign', 'malignant']
breakhist_benign_label = ['adenosis','fibroadenoma','phyllodes_tumor','tubular_adenoma']
breakhist_malignant_label = ['ductal_carcinoma','lobular_carcinoma','mucinous_carcinoma','papillary_carcinoma']
isic_label = ['isic_benign', 'isic_malignant']
idrid_label = ['idrid_negative', 'idrid_positive']

def decode_label(encoded):
  main_class = encoded[0]
  main_class = np.argmax(main_class) 
  child_class = np.argmax(encoded[1])  
  actual_label = 'unknown'
  if main_class == 0:
    if child_class == 0:
      actual_label = breakhist_benign_label[np.argmax(encoded[2])]
    elif child_class == 1:
     actual_label = breakhist_malignant_label[np.argmax(encoded[2])]
  elif main_class == 1:
    actual_label = idrid_label[child_class]
  elif main_class == 2:
    actual_label = isic_label[child_class]
  elif main_class == 3:
    actual_label = mias_label[child_class]

  return actual_label


def show_data(query_path):
  global class_result
  actual_label = decode_label(get_class(query_path))
  class_result = actual_label
  if actual_label == 'adenosis':
    return show_retrieved_images(query_path, breakhist_adenosis_path, adenosis_features, adenosis_scaler, adenosis_kmeans, 300)

  elif actual_label == 'fibroadenoma':
    return show_retrieved_images(query_path, breakhist_fibroadenoma_path, fibroadenoma_features, fibroadenoma_scaler, fibroadenoma_kmeans, 300)

  elif actual_label == 'phyllodes_tumor':
    return show_retrieved_images(query_path, breakhist_phyllodes_tumor_path, phyllodes_tumor_features, 
                          phyllodes_tumor_scaler, phyllodes_tumor_kmeans, 300)

  elif actual_label == 'tubular_adenoma':
    return show_retrieved_images(query_path, breakhist_tubular_adenoma_path, tubular_adenoma_features, 
                          tubular_adenoma_scaler, tubular_adenoma_kmeans, 300)

  elif actual_label =='ductal_carcinoma':
    return show_retrieved_images(query_path, breakhist_ductal_carcinoma_path, ductal_carcinoma_features, 
                          ductal_carcinoma_scaler, ductal_carcinoma_kmeans, 1024)
    
  elif actual_label =='lobular_carcinoma':
    return show_retrieved_images(query_path, breakhist_lobular_carcinoma_path, lobular_carcinoma_features,
                          lobular_carcinoma_scaler,lobular_carcinoma_kmeans, 1024)
    
  elif actual_label =='mucinous_carcinoma':
    return show_retrieved_images(query_path, breakhist_mucinous_carcinoma_path, mucinous_carcinoma_features, 
                          mucinous_carcinoma_scaler, mucinous_carcinoma_kmeans, 1024)
    
  elif actual_label =='papillary_carcinoma':
    return show_retrieved_images(query_path, breakhist_papillary_carcinoma_path, papillary_carcinoma_features, 
                          papillary_carcinoma_scaler, papillary_carcinoma_kmeans, 1024)

  elif actual_label == 'idrid_positive':
    return show_retrieved_images(query_path, idrid_positive_path, idrid_positive_features, 
                          idrid_positive_scaler, idrid_positive_kmeans, 100)

  elif actual_label == 'idrid_negative':
    return show_retrieved_images(query_path, idrid_negative_path, idrid_negative_features, 
                          idrid_negative_scaler, idrid_negative_kmeans, 100)

  elif actual_label == 'mias_benign':
    return show_retrieved_images(query_path, mias_benign_path, mias_benign_features, 
                          mias_benign_scaler, mias_benign_kmeans, 300)

  elif actual_label == 'mias_malignant':
    return show_retrieved_images(query_path, mias_malignant_path, mias_malignant_features, mias_malignant_scaler,
                          mias_malignant_kmeans, 300)

  elif actual_label == 'isic_benign':
    return show_retrieved_images(query_path, isic_benign_path, isic_benign_features, isic_benign_scaler, isic_benign_kmeans, 10)

  elif actual_label == 'isic_malignant':
    return show_retrieved_images(query_path, isic_malignant_path, isic_malignant_features, isic_malignant_scaler, isic_malignant_kmeans, 10)

def get_ground_truth(query_path):
  # for query_path in path:
  ground_truth = query_path.split('\\')[-2].replace('-','_')
  ground_truth = 'idrid_negative' if ground_truth=='nosymptoms' else 'idrid_positive' if ground_truth=='symptoms' else ground_truth

  return ground_truth
    
def exec_eval(query_path, image_paths, features, scaler, kmeans, numWords, relevant_count=10):
  query_features = get_query_features(kmeans, query_path, numWords, scaler)
  results = perform_search(query_features, features)
  images, path, distances, correct_id, precision, recall = [], [], [], [], [], []
  count, correct = 0, 0
  
  ground_truth = get_ground_truth(query_path)
  for (d, j) in results:   
    if (get_ground_truth(image_paths[j])) == ground_truth:
      correct = correct + 1      
      correct_id.append(1)
    else:
      correct_id.append(0)
    count = count + 1
    # print(count)
    precision.append(correct/count)
    recall.append(correct/relevant_count)
  return precision, correct_id, recall

def eval_query(query_path, relevant_count=10):
  query = decode_label(get_class(query_path))
  if query == 'adenosis':
    return exec_eval(query_path, breakhist_adenosis_path, adenosis_features, adenosis_scaler, adenosis_kmeans, 300)

  elif query == 'fibroadenoma':
    return exec_eval(query_path, breakhist_fibroadenoma_path, fibroadenoma_features, fibroadenoma_scaler, fibroadenoma_kmeans, 300)

  elif query == 'phyllodes_tumor':
    return exec_eval(query_path, breakhist_phyllodes_tumor_path, phyllodes_tumor_features, 
                          phyllodes_tumor_scaler, phyllodes_tumor_kmeans, 300)

  elif query == 'tubular_adenoma':
    return exec_eval(query_path, breakhist_tubular_adenoma_path, tubular_adenoma_features, 
                          tubular_adenoma_scaler, tubular_adenoma_kmeans, 300)

  elif query =='ductal_carcinoma':
    return exec_eval(query_path, breakhist_ductal_carcinoma_path, ductal_carcinoma_features, 
                          ductal_carcinoma_scaler, ductal_carcinoma_kmeans, 1024)
    
  elif query =='lobular_carcinoma':
    return exec_eval(query_path, breakhist_lobular_carcinoma_path, lobular_carcinoma_features,
                          lobular_carcinoma_scaler,lobular_carcinoma_kmeans, 1024)
    
  elif query =='mucinous_carcinoma':
    return exec_eval(query_path, breakhist_mucinous_carcinoma_path, mucinous_carcinoma_features, 
                          mucinous_carcinoma_scaler, mucinous_carcinoma_kmeans, 1024)
    
  elif query =='papillary_carcinoma':
    return exec_eval(query_path, breakhist_papillary_carcinoma_path, papillary_carcinoma_features, 
                          papillary_carcinoma_scaler, papillary_carcinoma_kmeans, 1024)

  elif query == 'idrid_positive':
    return exec_eval(query_path, idrid_positive_path, idrid_positive_features, 
                          idrid_positive_scaler, idrid_positive_kmeans, 100)

  elif query == 'idrid_negative':
    return exec_eval(query_path, idrid_negative_path, idrid_negative_features, 
                          idrid_negative_scaler, idrid_negative_kmeans, 100)

  elif query == 'mias_benign':
    return exec_eval(query_path, mias_benign_path, mias_benign_features, 
                          mias_benign_scaler, mias_benign_kmeans, 300)

  elif query == 'mias_malignant':
    return exec_eval(query_path, mias_malignant_path, mias_malignant_features, mias_malignant_scaler,
                          mias_malignant_kmeans, 300)

  elif query == 'isic_benign':
    return exec_eval(query_path, isic_benign_path, isic_benign_features, isic_benign_scaler, isic_benign_kmeans, 10)

  elif query == 'isic_malignant':
    return exec_eval(query_path, isic_malignant_path, isic_malignant_features, isic_malignant_scaler, isic_malignant_kmeans, 10)



@app.route('/image-retrieval')
def index():
  result_after = []
  query_path = os.path.abspath(os.curdir + "/dataset/Mias Split/train/mias-benign/0.pgm")
  show_data(query_path)
  query_image = cv2.imread(query_path)
  cv2.imwrite(os.curdir + "/app/static/query_image.png",query_image)

  for i in range(len(images_result)):
    images_temporary = cv2.imread(images_result[i])
    cv2.imwrite(os.curdir + f"/app/static/result_image{i}.png",images_temporary)
    result_after.append(f"/static/result_image{i}.png")


  return render_template("result.html",result_class = class_result, query = "/static/query_image.png", image = result_after, showing_labels=showing_labels,distance_result=distance_result,complete_all=zip(result_after,showing_labels,distance_result))

