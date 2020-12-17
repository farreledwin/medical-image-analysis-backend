from flask import Blueprint
from app.controllers.ImageHandler import image_retrieval

image_blueprints = Blueprint('image_blueprint',__name__)

image_blueprints.add_url_rule('/image-retrieval',view_func=image_retrieval, methods=['POST'])