from flask import Flask, render_template, request, flash, redirect
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

def predict(values, model_path):
    model = load_model(model_path)
    values = np.asarray(values)
    return model.predict(values.reshape(-1, 1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, 'models/diabetes.h5')
        return render_template('predict.html', pred=pred)
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, 'models/breast_cancer.h5')
        return render_template('predict.html', pred=pred)
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, 'models/heart.h5')
        return render_template('predict.html', pred=pred)
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, 'models/kidney.h5')
        return render_template('predict.html', pred=pred)
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, 'models/liver.h5')
        return render_template('predict.html', pred=pred)
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    if request.method == 'POST':
        try:
            file = request.files.get('image')  # Retrieve the uploaded file
            if file is not None:
                img = Image.open(file)
                img = img.resize((128, 128))
                img = np.asarray(img)
                img = img.reshape((1, 128, 128, 3))
                img = img.astype(np.float64) / 255.0
                model = load_model("models/malaria_model.h5") #loading the model
                pred = model.predict(img)
                pred_class = 'Uninfected' if pred[0] > 0.5 else 'Parasitized'
                return render_template('malaria_predict.html', pred=pred_class)
            else:
                message = "Please upload an Image"
                return render_template('malaria.html', message=message)
        except Exception as e:
            message = "An error occurred while processing the image: {}".format(str(e))
            return render_template('malaria.html', message=message)
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    if request.method == 'POST':
        try:
            file = request.files.get('image')  # Retrieve the uploaded file
            if file is not None:
                img = Image.open(file)
                img = img.resize((128, 128))  # Resize the image to the desired dimensions
                img = img.convert('RGB')  # Ensure the image is in RGB mode
                img = np.asarray(img)
                if img.shape != (128, 128, 3):
                    raise ValueError("Image has incorrect shape after resizing")
                img = img.reshape((1, 128, 128, 3))
                img = img.astype(np.float32) /255.0
                model = load_model("models/pneumonia_model.h5")
                pred = model.predict(img)
                pred_class = 'pneumonia' if pred[0] > 0.5 else 'Normal'
                return render_template('pneumonia_predict.html', pred=pred_class)
            else:
                message = "Please upload an Image"
                return render_template('pneumonia.html', message=message)
        except Exception as e:
            message = "An error occurred while processing the image: {}".format(str(e))
            return render_template('pneumonia.html', message=message)
    return render_template('pneumonia.html')

if __name__ == '__main__':
    app.run(debug=True)
