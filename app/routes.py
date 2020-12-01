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
from app import app
from flask import jsonify
from flask import request
import shutil
from flask_cors import CORS,cross_origin
from PIL import Image # No need for ImageChops
import math
from skimage import img_as_float
from skimage.metrics import mean_squared_error as mse
import staintools
# import skimage.metrics.mean_squared_error as mse

train_datas, test_datas = [], []

result_image = []
result_relevant = []
result_distances = []
clahe_image= ""

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


def getDescriptors(sift, img):
    kp, des = sift.detectAndCompute(img, None)
    return des

def readImage(img_path):
    global clahe_image
    img = cv2.imread(img_path,0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    cv2.imwrite(os.path.abspath(os.curdir +"/uploads/clahe.jpg"),img)
    with open(os.path.abspath(os.curdir +"/uploads/clahe.jpg"), "rb") as img_file:
        b64_string = base64.b64encode(img_file.read())
        clahe_image = b64_string.decode('utf-8')
    return cv2.resize(img,(300,300))

def readImageNoClahe(img_path):
    img = cv2.imread(img_path,1)
    return cv2.rezise(img, (300,300))

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
      print(i)
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


def predict(kmeans, model, input_data, num_words, scaler):
  #top class
  descriptor_list = []
  img = readImage(input_data)
  des = getDescriptors(sift, img)
  descriptor_list.append(des)
  im_features = extractFeatures(kmeans, descriptor_list, 1, num_words)
  im_features = scaler.transform(im_features)
  res = np.argmax(model.predict(im_features))
  if res == 0:
    #benign
    image = cv2.imread(input_data,1)
    descriptor_list = []
    des = getDescriptors(sift, image)
    descriptor_list.append(des)
    im_features = extractFeatures(benign_kmeans, descriptor_list, 1, 300)
    im_features = benign_scale.transform(im_features)
    res = np.argmax(benign_model.predict(im_features))
    new_input = benign_model.input
    new_output = benign_model.layers[-3].output
    nmodel = tf.keras.Model(new_input, new_output)
    res = nmodel.predict(im_features)
    return 'benign', res
  else :
    #mal
    image = cv2.imread(input_data, 1)
    descriptor_list = []
    des = getDescriptors(sift, image)
    descriptor_list.append(des)
    im_features = extractFeatures(malignant_kmeans, descriptor_list, 1, 300)
    im_features = malignant_scale.transform(im_features)
    res = np.argmax(malignant_model.predict(im_features))
    new_input = malignant_model.input
    new_output = malignant_model.layers[-3].output
    nmodel = tf.keras.Model(new_input, new_output)
    res = nmodel.predict(im_features)
    return 'malignant', res
    
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

# for i, image_path in enumerate(x_train):
#   if adenosis == 10 and fibroadenoma == 10 and phyllodes_tumor == 10 and tubular_adenoma == 10 and ductal_carcinoma == 10 and lobular_carcinoma == 10 and mucinous_carcinoma == 10 and papillary_carcinoma == 10:
#     break
#   image = readImage(image_path)
#   print ("Extract SIFT of %s image, %d of %d images" %(y_train[i], i, len(x_train)))
#   des = getDescriptors(sift, image)
#   if des is None:
#     des = np.zeros([1, 128])
#   repo_class, repo_features = predict(kmeans,model,image_path,num_words,scale)
#   print("kelasnya adalah "+repo_class)
#   if repo_class == 'benign':
#       if adenosis != 10 and y_train[i] == 'adenosis':
#           print("masuk ade")
#           b_features_repository.append(repo_features)
#           b_repository.append(image_path)
#           b_descriptor_list.append(des)
#           b_labels.append(y_train[i])
#           adenosis+=1
#           b_count+=1
#       elif fibroadenoma != 10 and y_train[i] == 'fibroadenoma':
#           b_features_repository.append(repo_features)
#           b_repository.append(image_path)
#           b_descriptor_list.append(des)
#           b_labels.append(y_train[i])
#           fibroadenoma+=1
#           b_count+=1
#       elif phyllodes_tumor != 10 and y_train[i] == 'phyllodes_tumor':
#           b_features_repository.append(repo_features)
#           b_repository.append(image_path)
#           b_descriptor_list.append(des)
#           b_labels.append(y_train[i])
#           phyllodes_tumor+=1
#           b_count+=1
#       elif tubular_adenoma !=10 and y_train[i] == 'tubular_adenoma':
#           b_features_repository.append(repo_features)
#           b_repository.append(image_path)
#           b_descriptor_list.append(des)
#           b_labels.append(y_train[i])
#           tubular_adenoma+=1
#           b_count+=1
#   else:
#       if ductal_carcinoma != 10 and y_train[i] == 'ductal_carcinoma':
#         m_features_repository.append(repo_features)
#         m_repository.append(image_path)
#         m_descriptor_list.append(des)
#         m_labels.append(y_train[i])
#         ductal_carcinoma+=1
#         m_count+=1
#       elif lobular_carcinoma !=10 and y_train[i] == 'lobular_carcinoma':
#         m_features_repository.append(repo_features)
#         m_repository.append(image_path)
#         m_descriptor_list.append(des)
#         m_labels.append(y_train[i])
#         lobular_carcinoma+=1
#         m_count+=1
#       elif mucinous_carcinoma !=10 and y_train[i] == 'mucinous_carcinoma':
#         m_features_repository.append(repo_features)
#         m_repository.append(image_path)
#         m_descriptor_list.append(des)
#         m_labels.append(y_train[i])
#         mucinous_carcinoma+=1
#         m_count+=1
#       elif papillary_carcinoma !=10 and y_train[i] == 'papillary_carcinoma':
#         m_features_repository.append(repo_features)
#         m_repository.append(image_path)
#         m_descriptor_list.append(des)
#         m_labels.append(y_train[i])
#         papillary_carcinoma+=1
#         m_count+=1

      

# print("sudah 10 semuanya")
# b_im_features = extractFeatures(benign_kmeans, b_descriptor_list, b_count, 300)
# m_im_features = extractFeatures(malignant_kmeans, m_descriptor_list, m_count, 300)
# b_im_features = benign_scale.transform(b_im_features)
# m_im_features = malignant_scale.transform(m_im_features)




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
# with open('b_descriptor_list.pickle', 'wb') as f:
#   pickle.dump(b_descriptor_list, f)

# with open('b_features_repository.pickle', 'wb') as f:
#   pickle.dump(b_features_repository,f)

# with open('b_im_features.pickle','wb') as f:
#   pickle.dump(b_im_features,f)

# with open('b_knn.pickle','wb') as f:
#   pickle.dump(b_knn,f) 

# with open('b_labels.pickle','wb') as f:
#   pickle.dump(b_labels,f)

# with open('b_repository.pickle','wb') as f:
#   pickle.dump(b_repository,f)

# with open('m_descriptor_list.pickle','wb') as f:
#   pickle.dump(m_descriptor_list,f)

# with open('m_features_repository.pickle','wb') as f:
#   pickle.dump(m_features_repository,f)

# with open('m_im_features.pickle','wb') as f:
#   pickle.dump(m_im_features,f)

# with open('m_knn.pickle','wb') as f:
#   pickle.dump(m_knn,f)

# with open('m_labels.pickle','wb') as f:
#   pickle.dump(m_labels,f)

# with open('m_repository.pickle','wb') as f:
#   pickle.dump(m_repository,f)



def cosine_similarity(a, b):
  return dot(a, b)/(norm(a)*norm(b))

def euclidean(a, b):
	return np.linalg.norm(np.array(a) - np.array(b))
 
def get_query_features(kmeans, input_data, num_words, scaler):
  descriptor_list = []
  
  img = readImage(input_data)
  des = getDescriptors(sift, img)
  descriptor_list.append(des)
  im_features = extractFeatures(kmeans, descriptor_list, 1, num_words)
  im_features = scaler.transform(im_features)
  return im_features

def perform_search(query_features, query_voc_features, feats, repo_voc_features, knn, max_results=10):
  temps = []

  for i in range(0, len(feats)):
    d = euclidean(query_features, feats[i])
    # d = (d + euclidean(query_color_features, c
    voc_d = euclidean(query_voc_features, repo_voc_features[i])/100
    knn_d = euclidean(knn.predict(query_voc_features), knn.predict(repo_voc_features[i].reshape(1,-1)))
    temps.append((d, i))
    temps = sorted(temps)

  results = []
  for temp in temps:
    d, i = temp
    results.append((d,i))
  # return results[:10], count
  return results


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

def show_retrieved_images(query_path, repositories, labels, scaler, kmeans, query_label, model,num_words= 50):
  global result_image
  global result_relevant
  global result_distances
  result_image = []
  result_relevant = []
  result_distances = []

  predicted_label, predicted_features = predict(kmeans, model, query_path, num_words, scaler)
  query_features = get_query_features(kmeans, query_path, num_words, scaler)
  if predicted_label == 'benign':
    query_features = get_query_features(benign_kmeans, query_path, 300, benign_scale)
    results = perform_search(predicted_features,query_features, b_features_repository, b_im_features, b_knn)
    repositories = repositories[0]
    labels = labels[0]
  else:    
    query_features = get_query_features(malignant_kmeans, query_path, 300, malignant_scale)
    results = perform_search(predicted_features,query_features, m_features_repository, m_im_features, m_knn)
    repositories = repositories[1]
    labels = labels[1]
  precision, recall = [], []
  correct, count = 0, 0
  images = []
  label = []
  distances = []
  path_image = []
  for (d, j) in results:
    # print(repositories[j])
    count = count + 1
    print(count)
    image = cv2.imread(repositories[j])
    label.append(labels[j])
    if query_label == labels[j]:
      correct = correct + 1   

    precision.append(correct/count)
    curr_recall = correct/10
    recall.append(curr_recall)
    #buat ganti path kalo pake WSL kan linux jadi rada beda
    edited_string = repositories[j].replace("\\","/")
    edited_string = edited_string.replace(edited_string[:3],'/mnt/d/')
    #end
    print(edited_string)
    path_image.append(edited_string)
    images.append(image)
    distances.append((d))
    if curr_recall == 1:
      break
  print(f"Query:")
  print(f"Actual Class : {query_label}")
  plt.axis('off')
  img = cv2.imread(query_path)#readImage(query_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # plt.imshow(img, cmap='gray')#cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB))
  # plt.show()
  # fig=plt.figure(figsize=(20, 8))
  columns = 5
  rows = 2
  for i in range(1, columns*rows + 1):
    # count = count + 1
    # if query_label == label[i-1]:
    #   correct = correct + 1        
    img = images[i-1]
    print("ffffffffeeeeee")
    print(path_image[i-1])
    with open(path_image[i-1], "rb") as img_file:
          b64_string = base64.b64encode(img_file.read())
          result_image.append(b64_string.decode('utf-8'))
   
  #   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  #   ax = fig.add_subplot(rows, columns, i)
    showing_label = "Relevant" if query_label == label[i-1] else "Not Relevant"
    result_relevant.append(showing_label)
    result_distances.append(str(distances[i-1]))
  #   precision.append(correct/count)
  #   recall.append(correct/10) # 10 itu relevant count
  #   ax.set_title(f"{label[i-1]}\n {distances[i-1]}")
  #   ax.set_title(f"{showing_label} \n {distances[i-1][0]} \n {distances[i-1][1]}")
  #   plt.axis('off')
  #   plt.imshow(img, cmap='gray')
    
  # plt.show()

  avg_prec = sum(precision)/count
  recall_11, precision_11 = [], []
  score = 1
  for i, r in enumerate(recall):
    if r*10 == score:
      recall_11.append(r)
      precision_11.append(precision[i])
      score = score + 1
    if score == 11:
      break
  return recall_11, precision_11, avg_prec, correct, result_image,result_relevant,result_distances,label


  ##################################### IMAGE REGISTRATION #############################################
image_regist_result = ""
def get_random_crop(image, crop_height, crop_width):

    max_x1 = image.shape[1] - crop_width
    max_y1 = image.shape[0] - crop_height

    x1 = 0#np.random.randint(0, max_x1)
    y1 = 0#np.random.randint(0, max_y1)

    img1 = image[y1: y1 + crop_height, x1: x1 + crop_width]
    max_x2 = image.shape[1] - crop_width
    max_y2 = image.shape[0] - crop_height

    x2 = 40 #np.random.randint(0, max_x2)
    y2 = 30 #np.random.randint(0, max_y2)

    img2 = image[y2: y2 + crop_height, x2: x2 + crop_width]
    return img1, img2, x1, x2, y1, y2


def rmsdiff(im1, im2):
    """Calculates the root mean square error (RSME) between two images"""
    return math.sqrt(mse(img_as_float(im1), img_as_float(im2)))

def get_stitched_image(img1, img2, M):
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

# Find SIFT or SURF and return Homography Matrix
def get_homography(img1, img2, algo, is_clahe):
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
    exit()


def registration(reference_path, target_path, algo='sift', is_clahe=0):
	# Get input set of images
  global image_regist_result
  img1 = cv2.imread(reference_path)
  img2 = cv2.imread(target_path)
  # img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)

  # Use SIFT to find keypoints and return homography matrix
  M =  get_homography(img1, img2, algo, is_clahe)

  # Stitch the images together using homography matrix
  result_image, theta, tx, ty, scale_x, scale_y = get_stitched_image(img2, img1, M)
  rmse = rmsdiff(img1, cv2.resize(result_image, (img1.shape[1], img1.shape[0])))
  cv2.imwrite(os.path.abspath(os.curdir +"/uploads/result.png"),result_image)

  with open(os.path.abspath(os.curdir +"/uploads/result.png"), "rb") as img_file:
      b64_string = base64.b64encode(img_file.read())
      image_regist_result = b64_string.decode('utf-8')
  
  print(f"RMSE: {rmse}")
  print(f"tx: {tx}, ty: {ty}, theta: {theta}")
  print(f"theta: {theta}")

  return rmse,tx,ty,theta


@app.route('/image-retrieval',methods=['POST'])
@cross_origin()
def index():
    test_recall, test_precision, test_avg_precision = [], [], []
    query_image = ""

    request.files['image'].save(os.path.abspath(os.curdir + "/uploads/"+str(request.files['image'].filename)))

    with open(os.path.abspath(os.curdir + "/uploads/"+str(request.files['image'].filename)), "rb") as img_file:
        b64_string = base64.b64encode(img_file.read())
        query_image = b64_string.decode('utf-8')

    recall, precision, avg_precision, correct, result_image, result_relevant, result_distances,label = show_retrieved_images(os.path.abspath(os.curdir + "/uploads/"+request.files['image'].filename),[b_repository, m_repository], 
                                                                      [b_labels, m_labels], scale, kmeans, 'adenosis', model,800)
    print(recall)
    print(precision)
    print(recall)
    test_recall.append(recall)
    test_precision.append(precision)
    test_avg_precision.append(avg_precision)
    # classes_avg_precision[label].append(avg_precision)
    # classes_precision[label].append(precision)
    # classes_recall[label].append(recall)
    # print(correct)
    # print("test pathnya : "+test_path)
    # print(f"avg_precision: {avg_precision}")


    return jsonify({"image" :result_image, 'query_image': query_image, 'clahe_image': clahe_image, 'label': label, 'distances_result': result_distances, 'average_precision': avg_precision})

@app.route('/image-registration',methods=['POST'])
@cross_origin()
def image_regis():
  ref_path = ''
  trg_path = ''
  ref_image = ''
  trg_image = ''
  calculation = None
  request.files['image_reference'].save(os.path.abspath(os.curdir + "/uploads/"+"reference-0.png"))
  ref_path = os.path.abspath(os.curdir +"/uploads/reference-0.png")

  with open(ref_path, "rb") as img_file:
      b64_string = base64.b64encode(img_file.read())
      ref_image = b64_string.decode('utf-8')

  request.files['image_target'].save(os.path.abspath(os.curdir + "/uploads/"+"target-0.png"))
  trg_path = os.path.abspath(os.curdir +"/uploads/target-0.png")

  with open(trg_path, "rb") as img_file:
      b64_string = base64.b64encode(img_file.read())
      trg_image = b64_string.decode('utf-8')

  if request.files['image_reference'] and request.files['image_target'] != 0:
    rmse,tx,ty,theta = registration(ref_path, trg_path, 'sift', 1)
    calculation = {
      "rmse" : rmse,
      "tx" : tx,
      "ty" : ty,
      "theta": theta
    }

  else:
    return jsonify({"message" : "Must Upload 2 Image (Reference Image and Target Image)"})
  
  
  return jsonify({"image_reference": ref_image, "image_target": trg_image, "result_image": image_regist_result,"calculate": calculation})

@app.route('/image-processing',methods=['POST'])
@cross_origin()
def image_process():

    if request.form.get('valueBtn') == "CLAHE":
      request.files['image'].save(os.path.abspath(os.curdir + "/uploads/"+str(request.files['image'].filename)))
      with open(os.path.abspath(os.curdir + "/uploads/"+str(request.files['image'].filename)), "rb") as img_file:
        b64_string = base64.b64encode(img_file.read())
        upload_image = b64_string.decode('utf-8')
      readImage(os.path.abspath(os.curdir + "/uploads/"+str(request.files['image'].filename)))
      return jsonify({"result_image": clahe_image,"upload_image": upload_image})

    elif request.form.get('valueBtn') == "Stain Normalization":
  
      request.files['reference_image'].save(os.path.abspath(os.curdir + "/uploads/"+str(request.files['reference_image'].filename)))
      # source_image = readImageNoClahe(os.path.abspath(os.curdir + "/uploads/"+str(request.files['reference_image'].filename)))

      request.files['target_image'].save(os.path.abspath(os.curdir + "/uploads/"+str(request.files['target_image'].filename)))
      # target_image = readImageNoClahe(os.path.abspath(os.curdir + "/uploads/"+str(request.files['target_image'].filename)))
      target = staintools.read_image(os.path.abspath(os.curdir + "/uploads/"+str(request.files['target_image'].filename)))
      to_transform = staintools.read_image(os.path.abspath(os.curdir + "/uploads/"+str(request.files['reference_image'].filename)))  
     
      normalizer = staintools.StainNormalizer(method='vahadane')
      normalizer.fit(target)
      transformed = normalizer.transform(to_transform)
      img_transformed = cv2.resize(transformed,(300,300))
      cv2.imwrite(os.path.abspath(os.curdir +"/uploads/result_stain.jpg"),img_transformed)

      with open(os.path.abspath(os.curdir +"/uploads/result_stain.jpg"), "rb") as img_file:
          b64_string = base64.b64encode(img_file.read())
          img_result_stain_b64 = b64_string.decode('utf-8')

      return jsonify({"result_image": img_result_stain_b64})

    return request.form.get('valueBtn')
