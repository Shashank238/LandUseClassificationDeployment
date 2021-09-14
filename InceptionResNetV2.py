import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionResNetV2
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
import matplotlib.pyplot as plt
IMAGE_SIZE = [331, 331]

train_path = 'Dataset/train'
valid_path = 'Dataset/test'

incep = InceptionResNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

for layer in incep.layers:
    layer.trainable = False


# useful for getting number of output classes
folders = glob('Dataset/train/*')
print(len(folders))

#FINE-TUNING THE BASE MODEL.
headModel = incep.output
headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(4032, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(4032, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
prediction= Dense(len(folders), activation="softmax")(headModel)
model_incep=  Model(inputs=incep.input, outputs=prediction)

model_incep.summary()


# tell the model what cost and optimization method to use
opt_2=Adam(lr=1e-4)
model_incep.compile(
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
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_InceptionResnetv2.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

callbacks = [
       tf.keras.callbacks.TensorBoard(log_dir='logs')]


# fit the model
# Run the cell. It will take some time to execute
result = model_incep.fit(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set),
  callbacks=[callbacks,checkpointer]
)




from tensorflow.keras.models import load_model

model=load_model('model_InceptionResnetv2_50e.h5')


plt.plot(result.history['loss'], label='train loss')
plt.plot(result.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(result.history['accuracy'], label='train acc')
plt.plot(result.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

test_score = model.evaluate_generator(test_set, 32)



print("[INFO] accuracy: {:.2f}%".format(test_score[1] * 100))

print("[INFO] Loss: ",test_score[0])

