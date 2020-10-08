from app import app
from keras.models import model_from_json
import os
from flask import jsonify
import numpy as np
from keras.preprocessing.image import img_to_array
import cv2
from sklearn.cluster import KMeans
from scipy.cluster.vq import *
import pickle
from sklearn.preprocessing import OneHotEncoder
from flask import render_template

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
json_file = open(dir_path + '/' +'model_ISIC.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(dir_path + '/' +"model_ISIC_weights.h5")
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


x_train = 0
x_test = 0
y_test = 0 
y_train = 0

def load_pickle():
    global x_train,x_test,y_test,y_train
    with open(os.path.realpath(os.curdir + "/pickle/ISIC_224_training_y_train.pickle"), 'rb') as f: 
        y_train = pickle.load(f)
    with open(os.path.realpath(os.curdir + "/pickle/ISIC_224_testing_x_test.pickle"), 'rb') as f: 
        x_test = pickle.load(f)
    with open(os.path.realpath(os.curdir + "/pickle/ISIC_224_testing_y_test.pickle"), 'rb') as f: 
        y_test = pickle.load(f)

    encoder = OneHotEncoder(sparse=False)
    y_train = y_train.reshape(len(y_train), 1)
    y_train = encoder.fit_transform(y_train)
    y_test = y_test.reshape(len(y_test),1)
    y_test = encoder.fit_transform(y_test)

def gen_sift_features(gray_img):
    
    sift = cv2.xfeatures2d.SIFT_create()
    # gray_img = cv2.resize(gray_img, (400, 400))
    kp, desc = sift.detectAndCompute(gray_img,  None)  
    
    return kp, desc

def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))


def works(new_image):
    feature = []
    desc_list = []
    label = [] 

    gray = new_image

    kp, desc= gen_sift_features(gray)
    global y_train
    print(len(y_train))
    for image_index in range(len(gray)):
        print(image_index)
        if desc is None:
            desc = np.zeros((1,128))
        label.append(y_train[image_index])
        desc_list.append((y_train[image_index], desc)) 

        descriptors = desc_list[0][1]
        for image_path, descriptor in desc_list[1:]:
            descriptors = np.vstack((descriptors, descriptor))  

        descriptors = descriptors.astype(float)


    numWords = 224 

    print ("Start k-means: %d words, %d key points" %(numWords, descriptors.shape[0]))
    voc, variance = kmeans(descriptors, numWords, 1) 


    im_features = np.zeros((len(label), numWords), "float32")
    for i in range(len(label)):
        print(i)
        words, distance = vq(desc_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1

    x_train = im_features
    y_train = np.asarray(label)

    return x_train

def get_prediction():
    image = cv2.imread(dir_path + '/' +"1.jpg")
 
    x_train = works(image)
    # image = cv2.resize(image, (224, 224))
    # image - np.array(image)
    # image = np.expand_dims(image, axis=0)

    newXtest = x_train.reshape((x_train.shape[0],1,224))
    result = np.argmax(loaded_model.predict(newXtest), axis=-1)
    result_accuracy = loaded_model.predict(newXtest)

    return result, result_accuracy

load_pickle()

@app.route('/predict-isic')
def index():
    result, result_accuracy = get_prediction()
    if(result[0] == 1):
        return render_template("result.html",result = 'malignant',user_image = "malignant-2.jpg", accuracy=result_accuracy[0][0][1] *100)
    else:
        return render_template("result.html",result = 'benign',user_image = "8.jpg", accuracy=result_accuracy[0][0][0] *100)
