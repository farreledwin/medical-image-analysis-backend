from flask import Flask
from keras.models import model_from_json
import json
import os 
from keras.preprocessing.image import array_to_img, img_to_array
from flask_cors import CORS
from app.routes import image_blueprints
from app import controllers

    
app = Flask(__name__)
app.register_blueprint(image_blueprints)
CORS(app)
from app import routes