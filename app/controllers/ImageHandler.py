from flask import request, jsonify
from app.helpers.ImageRetrieval import ImageRetrieval
from app.helpers.ImageRegistration import ImageRegistration
from app.helpers.ImageEnhancement import ImageEnhancement
from app.helpers.ImageSegmentation import ImageSegmentation
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


def image_registration():
  obj = ImageRegistration()
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
    tx,ty,theta,image_regist_result = obj.registration(ref_path, trg_path, 'sift', 1)
    calculation = {
      "tx" : tx,
      "ty" : ty,
      "theta": theta
    }

  else:
    return jsonify({"message" : "Must Upload 2 Image (Reference Image and Target Image)"})
  
  
  return jsonify({"image_reference": ref_image, "image_target": trg_image, "result_image": image_regist_result,"calculate": calculation})

def imageEnhancement():
    obj = ImageEnhancement()
    clahe_image = ""
    ref_path = ""
    trg_path = ""
    if request.form.get('valueBtn') == "CLAHE":
        request.files['image'].save(os.path.abspath(os.curdir + "/uploads/"+str(request.files['image'].filename)))
        with open(os.path.abspath(os.curdir + "/uploads/"+str(request.files['image'].filename)), "rb") as img_file:
          b64_string = base64.b64encode(img_file.read())
          upload_image = b64_string.decode('utf-8')
          clahe_image = obj.claheImage(os.path.abspath(os.curdir + "/uploads/"+str(request.files['image'].filename)))
          return jsonify({"result_image": clahe_image,"upload_image": upload_image})

    elif request.form.get('valueBtn') == "Stain Normalization":
      request.files['reference_image'].save(os.path.abspath(os.curdir + "/uploads/"+str(request.files['reference_image'].filename)))

      request.files['target_image'].save(os.path.abspath(os.curdir + "/uploads/"+str(request.files['target_image'].filename)))
      trg_path = os.path.abspath(os.curdir + "/uploads/"+str(request.files['target_image'].filename))
      ref_path = os.path.abspath(os.curdir + "/uploads/"+str(request.files['reference_image'].filename))

      img_result_stain_b64 = obj.stainImage(trg_path,ref_path)

      return jsonify({"result_image": img_result_stain_b64})

def imageSegmentation():
    watershed_image = ""
    kmeans_image = ""
    obj = ImageSegmentation()
    request.files['image'].save(os.path.abspath(os.curdir + "/uploads/"+str(request.files['image'].filename)))
    with open(os.path.abspath(os.curdir + "/uploads/"+str(request.files['image'].filename)), "rb") as img_file:
      b64_string = base64.b64encode(img_file.read())
      upload_image = b64_string.decode('utf-8')
      image_path = os.path.abspath(os.curdir + "/uploads/"+str(request.files['image'].filename))
      
      if request.form.get('valueBtn') == "Watershed":
        watershed_image = obj.wathershed(image_path)
        return jsonify({"result_image": watershed_image})
      
      elif request.form.get('valueBtn') == "K-Means":
        clusters_amount =  int(request.form.get("clusters_count"))
        kmeans_image = obj.kmeans_segmentation(image_path,clusters_amount)
      return jsonify({"result_image": kmeans_image})
        
