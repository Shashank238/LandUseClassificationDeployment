import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
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

rv2 = ResNet152V2(input_shape=(224,224,3), weights='imagenet', include_top=False)

for layer in rv2.layers:
    layer.trainable = False


# useful for getting number of output classes
folders = glob('Dataset/train/*')
print(len(folders))

#FINE-TUNING THE BASE MODEL.
headModel = rv2.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(2084, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2084, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2084, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
prediction= Dense(len(folders), activation="softmax")(headModel)
model_rv2=  Model(inputs=rv2.input, outputs=prediction)

model_rv2.summary()


# tell the model what cost and optimization method to use
opt_2=Adam()
model_rv2.compile(
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
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_ResNet152V2.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
clr = CyclicLR(base_lr=0.00001, max_lr=0.0006,step_size=2000,mode='triangular2')



# fit the model
# Run the cell. It will take some time to execute
result = model_rv2.fit(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set),
  callbacks=[checkpointer]
)



from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model=load_model('model_for_vgg16.h5')



result = model.fit(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set),
  callbacks=[callbacks,checkpointer]
)







