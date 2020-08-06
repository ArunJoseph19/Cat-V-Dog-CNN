# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:01:34 2020

@author: A ARUN JOSEPHRAJ
"""

import numpy as np
from flask import Flask, render_template
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model('models/1')

@app.route('/')
def predict():
    '''
    For rendering results on HTML GUI
    '''
    from keras.preprocessing import image
    test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
    test_image = image.img_to_array(test_image) #converting it into an array
    test_image = np.expand_dims(test_image, axis = 0) #
    result = model.predict(test_image)
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'

    return render_template('index.html', pred=format(prediction))




if __name__ == "__main__":
    app.run(debug=True)