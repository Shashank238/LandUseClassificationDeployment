import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
#from keras.applications.vgg16 import VGG16
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
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'D:\Python\Land-use classification\CLR-master')

from clr_callback import *
IMAGE_SIZE = [224, 224]

train_path = 'Dataset/train'
valid_path = 'Dataset/test'

dn = DenseNet201(input_shape=(224,224,3), weights='imagenet', include_top=False)

for layer in dn.layers:
    layer.trainable = False


# useful for getting number of output classes
folders = glob('Dataset/train/*')
print(len(folders))

#FINE-TUNING THE BASE MODEL.
headModel = dn.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(2084, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2084, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2084, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
prediction= Dense(len(folders), activation="softmax")(headModel)
model_dn=  Model(inputs=dn.input, outputs=prediction)

model_dn.summary()


# tell the model what cost and optimization method to use
opt_2=Adam()
model_dn.compile(
  loss='categorical_crossentropy',
  optimizer=opt_2,
  metrics=['accuracy']
)

# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Dataset/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


test_set = test_datagen.flow_from_directory('Dataset/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')



#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_Densenet201(2).h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
clr = CyclicLR(base_lr=0.00001, max_lr=0.0001,step_size=2000,mode='triangular2')



# fit the model
# Run the cell. It will take some time to execute
result = model.fit(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set),
  callbacks=[checkpointer,clr]
)


from tensorflow.keras.models import load_model

model_dn.save('model_for_Densenet201_50e.h5')

plt.plot(result.history['loss'], label='train loss')
plt.plot(result.history['val_loss'], label='val loss')
plt.legend()
plt.show()


# plot the accuracy
plt.plot(result.history['accuracy'], label='train acc')
plt.plot(result.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()


test_set = test_datagen.flow_from_directory('Dataset/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode="sparse",
                                            shuffle=False)

model=load_model('model_for_Densenet201(2).h5')
from tensorflow.keras.preprocessing import image

cls=test_set.classes
test_name = []

for key in test_set.class_indices:

    test_name.append(key)
test_datagen
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

y_pred = model.predict(test_set)


import numpy as np
y_pred = np.argmax(y_pred, axis=1)



classi_report=classification_report(cls,y_pred,target_names=test_name,digits=4)

acc=accuracy_score(cls,y_pred)
acc
import seaborn as sns
cm = confusion_matrix(cls, y_pred)
plt.figure(figsize=(20,20))
sns.heatmap(cm, fmt='.0f', annot=True, linewidths=0.2, linecolor='purple')
plt.xlabel('predicted value')
plt.ylabel('Truth value')
plt.show()





