import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
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

IMAGE_SIZE = [224, 224]

train_path = 'Dataset/train'
valid_path = 'Dataset/test'

mobile = MobileNet(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in mobile.layers:
    layer.trainable = False


# useful for getting number of output classes
folders = glob('Dataset/train/*')
print(len(folders))

#FINE-TUNING THE BASE MODEL.
headModel = mobile.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
prediction= Dense(len(folders), activation="softmax")(headModel)
model_mobile=  Model(inputs=mobile.input, outputs=prediction)

model_mobile.summary()


# tell the model what cost and optimization method to use
epochs=50
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8


opt_2=SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)
model_mobile.compile(
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
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_mobilenetAdam.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)



def exp_decay(epoch):
    lrate = learning_rate * np.exp(-decay_rate*epoch)
    return lrate

# learning schedule callback
from keras.callbacks import LearningRateScheduler
lr_rate = LearningRateScheduler(exp_decay)
callbacks_list = [lr_rate]
# fit the model
# Run the cell. It will take some time to execute
result = model_mobile.fit(
  training_set,
  validation_data=test_set,
  epochs=epochs,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set),
  callbacks=[callbacks_list,checkpointer]
)








