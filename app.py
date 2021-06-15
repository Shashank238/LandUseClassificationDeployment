from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf
from PIL import Image
import base64
import io
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

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

# Define a flask app
app = Flask(__name__,template_folder='templates')

# Model saved with Keras model.save()
MODEL_PATH ='model_for_Densenet201(2).h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    a = preds[0]
    ind=np.argmax(a)
    resultss=names[ind]
    
    return resultss
    
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['inpuFile']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        im = Image.open(file_path)
        data = io.BytesIO()
        im.save(data, "JPEG")
        encoded_img_data = base64.b64encode(data.getvalue())
        
        # Get the in-memory info using below code line.
        data = io.BytesIO()
        
        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return render_template("predict.html",prediction=result ,img_data=encoded_img_data.decode('utf-8'))
       

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)








    
 