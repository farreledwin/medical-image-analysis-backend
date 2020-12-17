from flask import request, jsonify
from app.helpers.ImageRetrieval import ImageRetrieval
import os
import base64
from app.controllers import Helper
def image_retrieval():
    obj = ImageRetrieval()
    test_recall, test_precision, test_avg_precision = [], [], []
    query_image = ""

    request.files['image'].save(os.path.abspath(os.curdir + "/uploads/"+str(request.files['image'].filename)))

    with open(os.path.abspath(os.curdir + "/uploads/"+str(request.files['image'].filename)), "rb") as img_file:
        b64_string = base64.b64encode(img_file.read())
        query_image = b64_string.decode('utf-8')

    recall, precision, avg_precision, correct, result_image, result_relevant, result_distances,label = obj.show_retrieved_images(os.path.abspath(os.curdir + "/uploads/"+request.files['image'].filename),[Helper.b_repository, Helper.m_repository], 
                                                                      [Helper.b_labels, Helper.m_labels], Helper.scale, Helper.kmeans, 'adenosis', Helper.model,800)
    clahe_image = obj.readImageClahe(os.path.abspath(os.curdir + "/uploads/"+request.files['image'].filename))
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
