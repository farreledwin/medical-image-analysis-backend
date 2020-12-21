from flask import Blueprint
from app.controllers.ImageHandler import image_retrieval,image_registration

image_blueprints = Blueprint('image_blueprint',__name__)

image_blueprints.add_url_rule('/image-retrieval',view_func=image_retrieval, methods=['POST'])

image_blueprints.add_url_rule('/image-registration',view_func=image_registration, methods=['POST'])
