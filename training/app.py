import sys
import os
import numpy as np
import cv2
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
# Model saved with Keras model.save()
MODEL_PATH ="./models/2/assets/models.h5"



custom_objects = {'CustomAdam': Adam}
model = load_model(MODEL_PATH,custom_objects=custom_objects,compile=False)

#model._make_predict_function()      
print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    # Load and preprocess the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = np.array(img) / 255.0  # Normalize the pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    preds = model.predict(img)

    return preds



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads',f.filename )  #secure_filename(f.filename)
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        pred_class = preds.argmax()              # Simple argmax
 
        
        class_names = ['Early_blight','Healthy','Late_blight']
        return class_names[pred_class]
    

        #return CATEGORIES[pred_class]
    return None


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


