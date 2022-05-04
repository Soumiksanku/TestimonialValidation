# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:20:01 2022

@author: Soumik Dhar
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from keras.preprocessing import image
import easyocr
reader = easyocr.Reader(['en'])


app = Flask(__name__)

original = cv2.imread("C:/Users/soumi/Downloads/Final Year Project 2/Final Year Project 2/jpeg/sample.jpg")
db = pd.read_csv("C:/Users/soumi/Downloads/Final Year Project 2/Final Year Project 2/data.csv")




def Certificate_validation(img_path):
  original_logo = original[100:700, 2450:3200]
  test = cv2.imread(img_path)
  output = reader.readtext(test)
  
  logo=test[100:700, 2450:3200]

  name = output[3][1]
  course = output[5][1]
  date = output[10][1]

  image_tensor_a= tf.convert_to_tensor(original_logo)
  image_tensor_b = tf.convert_to_tensor(logo)
  image_to_tensor_a = tf.keras.preprocessing.image.smart_resize(image_tensor_a,(224,224))
  image_to_tensor_b = tf.keras.preprocessing.image.smart_resize(image_tensor_b,(224,224))

  feature_of_image_a = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", trainable=True)([image_to_tensor_a])
  feature_of_image_b = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", trainable=False)([image_to_tensor_b])

  similarity = tf.keras.losses.cosine_similarity(
    [feature_of_image_a],
    [feature_of_image_b],
    axis=-1,
  )
    
  similarity_b = tf.math.reduce_sum(similarity[0])
  if similarity_b.numpy()<-0.8:
    count = 0
    nameFlag = 0
    courseFlag = 0
    dateFlag = 0
    for i in db['STUDENT NAME']:
      if(i.upper() == name.upper()):
        nameFlag = 1
        if(db.iloc[count]['COURSE TAKEN'].upper() == course.upper()):
          courseFlag = 1
          if(db.iloc[count]['DATE OF ISSUE'] == date):
            return ("Valid Certificate")
            dateFlag = 1
      count = count + 1
    if(nameFlag == 0):
      return ("Invalid Certificate, Name not Found")
    elif(courseFlag == 0):
      return ("Invalid Certificate, Course Mismatch")
    elif(dateFlag == 0):
      return ("Invalid Certificate, Issue Date Mismatch")
  else:
    return ("Invalid Certifiace, logo Mismatch")

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = Certificate_validation(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
    app.run(debug = True)

