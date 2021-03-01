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

class ImageRetrieval():

    result_image = []
    result_relevant = []
    result_distances = []
    clahe_image = ""
    sift = cv2.xfeatures2d.SIFT_create()

    def __init__(self):
        return None

    def predict(self,kmeans, model, input_data, num_words, scaler):
        #top class
        descriptor_list = []
        img = self.readImage(input_data)
        des = self.getDescriptors(self.sift, img)
        descriptor_list.append(des)
        im_features = self.extractFeatures(kmeans, descriptor_list, 1, num_words)
        im_features = scaler.transform(im_features)
        res = np.argmax(model.predict(im_features))
        if res == 0:
            #benign
            image = cv2.imread(input_data,1)
            descriptor_list = []
            des = self.getDescriptors(self.sift, image)
            descriptor_list.append(des)
            im_features = self.extractFeatures(Helper.benign_kmeans, descriptor_list, 1, 300)
            im_features = Helper.benign_scale.transform(im_features)
            res = np.argmax(Helper.benign_model.predict(im_features))
            new_input = Helper.benign_model.input
            new_output = Helper.benign_model.layers[-3].output
            nmodel = tf.keras.Model(new_input, new_output)
            res = nmodel.predict(im_features)
            return 'benign', res
        else :
            #mal
            image = cv2.imread(input_data, 1)
            descriptor_list = []
            des = self.getDescriptors(self.sift, image)
            descriptor_list.append(des)
            im_features = self.extractFeatures(Helper.malignant_kmeans, descriptor_list, 1, 300)
            im_features = Helper.malignant_scale.transform(im_features)
            res = np.argmax(Helper.malignant_model.predict(im_features))
            new_input = Helper.malignant_model.input
            new_output = Helper.malignant_model.layers[-3].output
            nmodel = tf.keras.Model(new_input, new_output)
            res = nmodel.predict(im_features)
            return 'malignant', res

    
    def getDescriptors(self,sift, img):
        kp, des = sift.detectAndCompute(img, None)
        return des
        
    
    def readImage(self,img_path):
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

    def readImageClahe(self,img_path):
        global clahe_image
        img = cv2.imread(img_path,0)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        cv2.imwrite(os.path.abspath(os.curdir +"/uploads/clahe.jpg"),img)
        with open(os.path.abspath(os.curdir +"/uploads/clahe.jpg"), "rb") as img_file:
            b64_string = base64.b64encode(img_file.read())
            clahe_image = b64_string.decode('utf-8')
        return clahe_image
    
    def readImageNoClahe(self,img_path):
        img = cv2.imread(img_path,1)
        return cv2.rezise(img, (300,300))
    
    def vstackDescriptors(self,descriptor_list):
        descriptors = np.array(descriptor_list[0])
        for descriptor in descriptor_list[1:]:
            descriptors = np.vstack((descriptors, descriptor)) 

        return descriptors
    
    def clusterDescriptors(self,descriptors, no_clusters):
        kmeans = MiniBatchKMeans(n_clusters = no_clusters).fit(descriptors)
        return kmeans
    
    def extractFeatures(self,kmeans, descriptor_list, image_count, no_clusters):
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
                continue                    # continue
        return im_features
    
    def normalizeFeatures(self,scale, features):
        return scale.transform(features)
    
    def plotHistogram(self,im_features, no_clusters):
        x_scalar = np.arange(no_clusters)
        y_scalar = np.array([abs(np.sum(im_features[:,h], dtype=np.int32)) for h in range(no_clusters)])

        plt.bar(x_scalar, y_scalar)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title("Complete Vocabulary Generated")
        plt.xticks(x_scalar + 0.4, x_scalar)
        plt.show()

    def cosine_similarity(self,a, b):
        return dot(a, b)/(norm(a)*norm(b))

    
    def euclidean(self,a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

        
    def get_query_features(self,kmeans, input_data, num_words, scaler):
        descriptor_list = []   
        img = self.readImage(input_data)
        des = self.getDescriptors(self.sift, img)
        descriptor_list.append(des)
        im_features = self.extractFeatures(kmeans, descriptor_list, 1, num_words)
        im_features = scaler.transform(im_features)
        return im_features

    
    def perform_search(self,query_features, query_voc_features, feats, repo_voc_features, knn, max_results=10):
        temps = []

        for i in range(0, len(feats)):
            d = self.euclidean(query_features, feats[i])
            # d = (d + euclidean(query_color_features, c
            voc_d = self.euclidean(query_voc_features, repo_voc_features[i])/100
            knn_d = self.euclidean(knn.predict(query_voc_features), knn.predict(repo_voc_features[i].reshape(1,-1)))
            temps.append((d, i))
            temps = sorted(temps)

        results = []
        for temp in temps:
            d, i = temp
            results.append((d,i))
        # return results[:10], count
        return results

    def show_retrieved_images(self,query_path, repositories, labels, scaler, kmeans, query_label, model,num_words= 50):
        global result_image
        global result_relevant
        global result_distances
        result_image = []
        result_relevant = []
        result_distances = []

        predicted_label, predicted_features = self.predict(kmeans, model, query_path, num_words, scaler)
        query_features = self.get_query_features(kmeans, query_path, num_words, scaler)
        if predicted_label == 'benign':
            query_features = self.get_query_features(Helper.benign_kmeans, query_path, 300, Helper.benign_scale)
            results = self.perform_search(predicted_features,query_features, Helper.b_features_repository, Helper.b_im_features, Helper.b_knn)
            repositories = repositories[0]
            labels = labels[0]
        else:    
            query_features = self.get_query_features(malignant_kmeans, query_path, 300, malignant_scale)
            results = self.perform_search(predicted_features,query_features, Helper.m_features_repository, Helper.m_im_features, Helper.m_knn)
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
            #buat ganti path kalo pake WSL kan linux jadi rada beda, harus replace soalnya kmrn bikin repo pake laptop farjun
            edited_string = repositories[j].replace("\\","/")
            edited_string = edited_string.replace(edited_string[:3],'/mnt/d/')
            edited_string = edited_string.replace("/mnt/d/Data Farrel/Kuliah +Ngajar/SKRIPSI/coding/skripsi-backend/","")
            #end
            print(edited_string)
            path_image.append(edited_string)
            images.append(image)
            distances.append((d))
            if curr_recall == 1:
                break
        print(f"Query:")
        print(f"Actual Class : {query_label}")
        # plt.axis('off')
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

    
