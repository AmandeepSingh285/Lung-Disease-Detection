import os
import base64
from xml.dom import xmlbuilder
import numpy as np
import io
# from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.models import load_model
# from keras.applications import ImageDataGenerator
from keras.utils import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image 
from pyexpat import model
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from datetime import datetime
from pathlib import Path
Path("./static").mkdir(parents=True, exist_ok=True)

UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app._static_folder = 'static'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model(model_name):
    print('./models/' + model_name + '.h5')
    model_name = load_model('./models/' + model_name + '.h5')
    return model_name

@app.route('/', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = 'input.jpeg'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(filepath) 
            file.save(filepath)
        
        models = request.form.getlist('model')
        print(models)
        if models == []:
            flash('No ML model selected')
            return redirect(request.url)

        MODELS = ['MobileNet', 'InceptionNetV3', 'VGG16', 'AlexNet']
        model_dict = {}
        for MODEL in MODELS:
            if MODEL in models:
                model_dict[MODEL] = True
            else:
                model_dict[MODEL] = False
        print(model_dict)
        
        pred = {}
        for model in model_dict:
            if model_dict[model]:
                transfer_model = get_model(model)

                # predicting images
                img = keras.utils.load_img('./static/input.jpeg', target_size=(224, 224))
                x = keras.utils.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = x/255

                pred[model] = transfer_model.predict(x)
            
        
        print(pred)
                
        return render_template('result.html', model_dict = model_dict, pred = pred)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)