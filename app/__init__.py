from flask import Flask
from keras.models import model_from_json
import json
import os 
from keras.preprocessing.image import array_to_img, img_to_array
from flask_cors import CORS

# dir_path = os.path.dirname(os.path.realpath(__file__))
# json_file = open(dir_path + '/' +'model_ISIC.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# dir_path = os.path.dirname(os.path.realpath(__file__))
# with open(dir_path + '/' + 'model_ISIC.json', 'r') as json_file:
#     model = load_model(json_file)
    
app = Flask(__name__)
CORS(app)

from app import routes