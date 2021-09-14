# Import all Necessary Library
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image
import sys
# Importing Cyclic Learning rate Library
sys.path.insert(1, 'D:\Python\Land-use classification\CLR-master')
from clr_callback import *

# Test dataset Path
valid_path = 'Dataset/test'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Use the Image Data Generator to import the images from the dataset
test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory('Dataset/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode="sparse",
                                            shuffle=False)

from tensorflow.keras.models import load_model

# Load the best model
model=load_model('model_for_Densenet201(2).h5')
from tensorflow.keras.preprocessing import image

# Getting the testing dataset labels
cls=test_set.classes

# Test dataset classes
for key in test_set.class_indices:
    test_name.append(key)
test_name

names=['airplane',
 'airport',
 'baseball_diamond',
 'basketball_court',
 'beach',
 'bridge',
 'chaparral',
 'christmas_tree_farm',
 'church',
 'circular_farmland',
 'closed_road',
 'cloud',
 'commercial_area',
 'crosswalk',
 'dense_residential',
 'desert',
 'football_field',
 'forest',
 'freeway',
 'golf_course',
 'ground_track_field',
 'harbor',
 'industrial_area',
 'intersection',
 'island',
 'lake',
 'meadow',
 'medium_residential',
 'mobile_home_park',
 'mountain',
 'oil_gas_field',
 'oil_well',
 'overpass',
 'palace',
 'parking_lot',
 'priority traffic',
 'railway',
 'railway_station',
 'rectangular_farmland',
 'river',
 'roundabout',
 'runway',
 'sea_ice',
 'ship',
 'shipping_yard',
 'signal traffic',
 'snowberg',
 'solar_panel',
 'sparse_residential',
 'stadium',
 'storage_tank',
 'tennis_court',
 'terrace',
 'thermal_power_station',
 'transformer_station',
 'wastewater_treatment_plant',
 'wetland']

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score

# Predicting the saved model on test dataset
y_pred = model.predict(test_set)

# Converting the one-hot code encoded to argmax
import numpy as np
y_pred = np.argmax(y_pred, axis=1)

# Calculating the accuracy of the model
acc=accuracy_score(cls,y_pred)
acc

# Calculating the classification report
classi_report=classification_report(cls,y_pred,target_names=names,digits=4)

# Calculating and plotting the confusion matrix on heat map
import seaborn as sns
cm = confusion_matrix(cls, y_pred)
plt.figure(figsize=(20,20))
sns.heatmap(cm, fmt='.0f', annot=True, linewidths=0.2, linecolor='purple')
plt.xlabel('predicted value')
plt.ylabel('Truth value')
plt.show()

# Testing on a random Image
img=image.load_img('Dataset/Test/desert/desert_013.jpg',target_size=(224,224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = np.array(x, 'float32')
x /= 255
preds = model.predict(x)
a = preds[0]
ind=np.argmax(a)
print('Prediction:', names[ind])
result=names[ind]





















