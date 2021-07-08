from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dropout, Flatten, Dense
from keras.optimizer_v2.adam import Adam

import os
import numpy as np
import h5py

weights_filename='weights/top_model_weights/fc_model_weights.hdf5'

def complete_model(weights_path=weights_filename):
    inc_model=InceptionV3(include_top=False,
                          weights='imagenet',
                          input_shape=((150, 150, 3)))

    x = Flatten()(inc_model.output)
    x = Dense(64, activation='relu', name='dense_one')(x)
    x = Dropout(0.5, name='dropout_one')(x)
    x = Dense(64, activation='relu', name='dense_two')(x)
    x = Dropout(0.5, name='dropout_two')(x)
    top_model=Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs=[inc_model.input], outputs=[top_model])
    model.load_weights(weights_filename, by_name=True)

    for layer in inc_model.layers[:205]:
        layer.trainable = False


    return model
if __name__ == "__main__":

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory('data/img_train/',
                                                        target_size=(150, 150),
                                                        batch_size=32,
                                                        class_mode='binary')

    val_generator = test_datagen.flow_from_directory('data/img_val/',
                                                        target_size=(150, 150),
                                                        batch_size=32,
                                                        class_mode='binary')

    pred_generator=test_datagen.flow_from_directory('data/img_val/',
                                                        target_size=(150,150),
                                                        batch_size=100,
                                                        class_mode='binary')

    epochs=int(input('How much epochs?:'))

    filepath="weights/complete_model_checkpoint_weights/weights-improvement.hdf5"

    if not os.path.exists('weights/complete_model_checkpoint_weights/'):
        os.makedirs('weights/complete_model_checkpoint_weights/')

    model=complete_model()

    model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
    model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=2000,
        verbose=1)

    model.save(filepath)

    loss, accuracy = model.evaluate_generator(pred_generator, steps=100)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
