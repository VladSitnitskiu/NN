from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
import os

inc_model=InceptionV3(include_top=False,
                      weights='imagenet', 
                      input_shape=((150, 150, 3)))

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory('data/img_train/',
                                                        target_size=(150, 150),
                                                        batch_size=32,
                                                        class_mode=None,
                                                        shuffle=False)

val_generator = datagen.flow_from_directory('data/img_val/',
                                                              target_size=(150, 150),
                                                              batch_size=32,
                                                              class_mode=None,
                                                              shuffle=False)

if not os.path.exists('datagened/'):
    os.makedirs('datagened/')

datagened_train = inc_model.predict_generator(train_generator, 2000)
np.save(open('datagened/datagened_train.npy', 'wb'), datagened_train)

datagened_val = inc_model.predict_generator(val_generator, 2000)
np.save(open('datagened/datagened_val.npy', 'wb'), datagened_val)
print('Finished')
