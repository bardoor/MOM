import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
import numpy as np
import dataset

def create_model(classes_count, dropout=0.2, base_units=16):
    if base_units < 4:
        raise Exception('base_units must be greater than 4')
    
    model = keras.Sequential([
        layers.Input(shape=(None, 34)),
        layers.LSTM(units=base_units, return_sequences=True),
        layers.LSTM(units=int(base_units/2)),
        layers.Dense(units=base_units),
        layers.Dropout(dropout),
        layers.Dense(units=int(base_units/2)),
        layers.Dropout(dropout),
        layers.Dense(units=classes_count, activation='softmax') 
    ])

    model.compile(
        loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
    )

    return model

def train_model(model, dataset_file, output_weights="best_model.h5"):
    (x_train, y_train), (x_test, y_test) = dataset.load_data(dataset_file)

    callbacks = [
        ModelCheckpoint(filepath=output_weights, monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(
        x_train, y_train,
        epochs=250,
        batch_size=32,
        validation_split=0.2,
        shuffle=False,
        callbacks=callbacks
    )

    return history
    
def predict(model, path_to_video, deviation_threshold=0.1):
    for predicion in dataset.generate(path_to_video):
        predicted_class = np.std(predicion) > deviation_threshold if tf.argmax(model(predicion)[0]) else None
        yield predicted_class

