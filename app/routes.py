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

train_datas, test_datas = [], []

result_image = []
result_relevant = []
result_distances = []

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
    img = cv2.imread(img_path,0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    return cv2.resize(img,(300,300))

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
    return 'benign', np.argmax(res)
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
    res = malignant_model.predict(im_features)
    return 'malignant', np.argmax(res)
    
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
    temps.append((d, i, voc_d, knn_d))
    temps = sorted(temps)

  results = []
  for temp in temps:
    d, i, voc_d, knn_d = temp
    results.append((d,i, voc_d, knn_d))
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
  for (d, j, voc_d, knn_d) in results:
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
    path_image.append(repositories[j])
    images.append(image)
    distances.append((d, voc_d))
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
    print(path_image[i-1])
    with open(path_image[i-1], "rb") as img_file:
          b64_string = base64.b64encode(img_file.read())
          result_image.append(b64_string.decode('utf-8'))
   
  #   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  #   ax = fig.add_subplot(rows, columns, i)
    showing_label = "Relevant" if query_label == label[i-1] else "Not Relevant"
    result_relevant.append(showing_label)
    result_distances.append(str(distances[i-1][0])+"-"+" "+str(distances[i-1][1]))
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
  return recall_11, precision_11, avg_prec, correct, result_image,result_relevant,result_distances

@app.route('/image-retrieval')
def index():
    test_recall, test_precision, test_avg_precision = [], [], []

    recall, precision, avg_precision, correct, result_image, result_relevant, result_distances = show_retrieved_images(os.path.abspath(os.curdir + "/dataset/benign/validation/phyllodes_tumor/phyllodes_tumor110.jpg"), 
                                                                      [b_repository, m_repository], 
                                                                      [b_labels, m_labels], scale, kmeans, 'phyllodes_tumor', model,800)
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


    return jsonify({"image" :result_image, 'relevant_result': result_relevant, 'distances_result': result_distances, 'average_precision': avg_precision})

