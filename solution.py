import pickle
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
import glob
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model('weights\complete_model_checkpoint_weights\weights-improvement.hdf5')


def predict(img):
    img = preprocess_input(np.array([img]))
    return (model.predict(img))
    

i = 0
for s in glob.glob("data\solute\*"):
    img = img_to_array(load_img(s, target_size=(150, 150)))
    label = predict(img)
    print(label)
    i+=1
    if label >0.5:
        os.rename(s, f"data/solute/{i}_{'dog'}.jpg")
    else:
        os.rename(s, f"data/solute/{i}_{'cat'}.jpg")