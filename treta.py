#!/usr/bin/env python
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

path = '.'
import psutil
import humanize
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.layers import AveragePooling2D, SeparableConv2D, Dropout, Flatten, Dense, Input, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from imutils import paths
import seaborn as sns
import pandas as pd
import numpy as np
import cv2

def show_image(img):
    '''
    Quick display of image in grayscale 
    '''
    plt.imshow(img, cmap = 'gray')
    plt.title('Example X-Ray scan')
    plt.grid(False)
    plt.axis('off')
    plt.show()

def plot_cm(labels, predictions):
    '''
    Plot the confusion matrix
    '''
    print(classification_report(labels, predictions,
	target_names=lb.classes_))

    cm = confusion_matrix(labels, predictions)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    
    print("accuracy: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))
    print()
    print('Correct Healthy Patient Detection (True Negatives): ', cm[0][0])
    print('Incorrect Covid-19 Detection (False Positives): ', cm[0][1])
    print('Incorrect Healthy Patient Detection (False Negatives): ', cm[1][0])
    print('Correct Covid-19 Detection (True Positives): ', cm[1][1])
    print('Total Patietns with Covid-19: ', np.sum(cm[1]))

    print()
    plt.figure(figsize=(7,7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

# In[19]:


df = pd.read_csv(path+'/metadata.csv')
df.head() # Show samples from Covid-19 dataset 


# In[20]:


df[['finding','view','modality','location']]


# In[21]:


# Initialize the variables for training
dataset_path = path+'/dataset'
init_lr = 1e-3
epochs = 100
batch_size = 5


# Read all files from path
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_path))
data = []
labels_ = []

# loop over the image paths
for imagePath in imagePaths:
    #print(imagePath)
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]
    #print(label)
    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print(image.shape)
    image = cv2.resize(image, (256, 256))

    # update the data and labels lists, respectively
    data.append(image)
    labels_.append(label)

# Normalize images to range [0,1]
data = np.array(data) / 255.0
labels_ = np.array(labels_)

# Plot example patient scan
print('Number of training images: ', len(data))
print()
show_image(data[0])

# perform one-hot encoding on the labels
print('Example scan label:\t ', labels_[0])
lb = LabelBinarizer()
labels = lb.fit_transform(labels_)
labels = to_categorical(labels)
print('One-hot encoded label: ', labels[0])


# In[22]:


# split training and test data
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=3, shuffle=True)

(trainX, validX, trainY, validY) = train_test_split(trainX, trainY,
	test_size=0.20, stratify=trainY, random_state=3, shuffle=True)

print('Number of training pairs: ', len(trainX))
print('Number of validation pairs: ', len(validX))
print('Number of testing pairs: ', len(testX))

# # initialize the training data augmentation object
trainAug = ImageDataGenerator(
	# rescale=1 / 255.0,
	rotation_range=10,
	# zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.05,
	# horizontal_flip=True,
	# vertical_flip=True,
	fill_mode="nearest")

# Load the VGG16 network
baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(256, 256, 3)))

# Make all pre-trained layers from VGG19 non-trainable 
for layer in baseModel.layers[:-3]:
    layer.trainable = False

# Add trainable fully-connected (FC) layers for predictions
newModel = baseModel.output
newModel = AveragePooling2D(pool_size=(4, 4))(newModel)
newModel = Flatten(name="flatten")(newModel)
newModel = Dense(64, activation="relu")(newModel)
newModel = Dropout(0.5)(newModel)
newModel = Dense(2, activation="softmax")(newModel)

# Stack the FC layers on top of VGG19 model
model = Model(inputs=baseModel.input, outputs=newModel, name='Covid19_Detector')

# compile our model
print("\n[INFO] compiling model...")
opt = Adam(lr=init_lr, decay=init_lr / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


# In[23]:


idx = np.random.randint(0,len(data))
print('idx: ', idx)
print()
print(labels_[idx])
print(labels[idx])


# In[24]:


class_weights = {0 : 1, 1 : 2.01}


# In[25]:


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    verbose=1,
    patience=15,
    mode='max',
    restore_best_weights=True)


# In[26]:


# train the model
print("[INFO] training new model ...")
results = model.fit_generator(
	trainAug.flow(trainX, trainY),
	# steps_per_epoch=len(trainX) // batch_size,
	validation_data=(validX, validY),
	# validation_steps=len(testX) // batch_size,
    callbacks = [early_stopping],
	epochs=epochs,
    # class_weight = class_weights
    )


# Analyse Results from the Test Set

# In[27]:


# Make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=batch_size)
print('\nNumber of test scans: ', len(testX))
print('Predicted class probabilities:')
print(predIdxs)

# Find the predicted labels 
predIdxs = np.argmax(predIdxs, axis=1)
print('\nPredicted outcome (Covid=1, Normal=0):')
print(predIdxs)
print('Ground-truth outcome:')
# print(testY)
trueIdxs = np.argmax(testY, axis=1)
print(trueIdxs)


# Loss/Accuracy Curve

# In[28]:


N = len(results.history["loss"])
plt.style.use("ggplot")
plt.figure(figsize = (20,10))
plt.plot(np.arange(0, N), results.history["loss"], label="train_loss", color = 'firebrick')
plt.plot(np.arange(0, N), results.history["val_loss"], label="val_loss", color = 'salmon')
plt.plot(np.arange(0, N), results.history["accuracy"], label="train_acc", color = 'teal')
plt.plot(np.arange(0, N), results.history["val_accuracy"], label="val_acc",color = 'cadetblue')
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig('loss_curve.pdf', format='pdf')
plt.show()


# compute the confusion matrix and and use it to derive the raw accuracy, sensitivity, and specificity

# In[29]:


predicted_metrics = model.evaluate(testX, testY,
                                  batch_size=batch_size, verbose=0)
for name, value in zip(model.metrics_names, predicted_metrics):
  print(name, ': ', value)
print()

plot_cm(trueIdxs, predIdxs)
model_json = model.to_json()
with open("modelo.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelo.h5")
print("Saved model to disk")
