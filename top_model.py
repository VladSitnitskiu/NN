from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense

import os
import numpy as np
import h5py

train_data = np.load(open('datagened/datagened_train.npy', 'rb'))
train_labels = np.array([0] * 1000 + [1] * 1000)

val_data = np.load(open('datagened/datagened_val.npy', 'rb'))
val_labels = np.array([0] * 1000 + [1] * 1000)

def fc_model():
    fc_model = Sequential()
    fc_model.add(Flatten(input_shape=train_data.shape[1:]))
    fc_model.add(Dense(64, activation='relu', name='dense_one'))
    fc_model.add(Dropout(0.5, name='dropout_one'))
    fc_model.add(Dense(64, activation='relu', name='dense_two'))
    fc_model.add(Dropout(0.5, name='dropout_two'))
    fc_model.add(Dense(1, activation='sigmoid', name='output'))

    return fc_model


epochs=int(input('How much epochs?:'))

model=fc_model()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs, validation_data=(val_data, val_labels))

if not os.path.exists('weights/top_model_weights/'):
    os.makedirs('weights/top_model_weights/')

model.save_weights('weights/top_model_weights/fc_model_weights.hdf5')

print('Finished')
print('-'*50)
loss, accuracy = model.evaluate(val_data, val_labels)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
