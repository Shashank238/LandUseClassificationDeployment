import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam, SGD , RMSprop
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image

IMAGE_SIZE = [331, 331]

train_path = 'Dataset/train'
valid_path = 'Dataset/test'

mob = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

for layer in mob.layers:
    layer.trainable = False


# useful for getting number of output classes
folders = glob('Dataset/train/*')
print(len(folders))

#FINE-TUNING THE BASE MODEL.
headModel = mob.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(4032, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(4032, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
prediction= Dense(len(folders), activation="softmax")(headModel)
model_mob=  Model(inputs=mob.input, outputs=prediction)

model_mob.summary()


# tell the model what cost and optimization method to use
opt_2=Adam(lr=1e-4)
model_mob.compile(
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
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_MobileNetV2.h5', verbose=1, save_best_only=True)

callbacks = [
       tf.keras.callbacks.TensorBoard(log_dir='logs')]


# fit the model
# Run the cell. It will take some time to execute
result = model_mob.fit(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set),
  callbacks=[callbacks,checkpointer]
)











