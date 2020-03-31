# -*- coding: utf-8 -*-
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

path = '.'
import psutil
import humanize
import os
import matplotlib.pyplot as plt
from imutils import paths
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.applications import preprocess_input
init_lr = 1e-3
epochs = 100
batch_size = 5

json_file = open('modelo.json', 'r')
loaded_model_json = json_file.read()
json_file.close()


loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelo.h5")
print("Loaded model from disk")

label = to_categorical('1')
image = cv2.imread("covid.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))
data  = np.array(np.expand_dims(image, axis=0))/255.0

predIdxs = [loaded_model.predict(data)[0][::-1]]
predIdxs = np.argmax(predIdxs, axis=1)
if predIdxs[0]== 1:
	print("tem covid-19")
else:
	print("não tem covid-19")

