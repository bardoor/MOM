import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
import numpy as np

import dataset
from keypoints import KeypointsLoader 


def create_model(classes_count, dropout=0.2, base_units=16):
    """
    Создаёт модель классификации движений на видео.

    Параметры:
        classes_count
            Число классов, которое необходимо спрогнозировать.

        dropout
            Вероятность дропаута для полносвязанных слоев.
        
        base_units
            Количество нейронов в первом слою сети, относительно которого считается
            количество нейронов в других слоях.
    
    Возвращает:
        Модель классификации движений на видео.
    """
    if base_units < 4:
        raise RuntimeError('base_units must be greater than 4')

    model = keras.Sequential([
        layers.Input(shape=(None, 34)),
        layers.LSTM(units=base_units, return_sequences=True),
        layers.LSTM(units=(base_units // 2)),
        layers.Dense(units=base_units),
        layers.Dropout(dropout),
        layers.Dense(units=(base_units // 2)),
        layers.Dropout(dropout),
        layers.Dense(units=classes_count, activation='softmax')
    ])

    # model.compile(
    #    loss='categorical_crossentropy',
    #    optimizer='adam',
    #    metrics=['accuracy']
    # )

    return model

def train_model(model, dataset_file, sample_size, shift, output_model, **kwargs):
    """
    Тренирует модель классификации движений на видео.

    Параметры:
        model
            Классифкационная модель.
        
        dataset_file
            Csv-файл датасета.
        
        sample_size
            Размер одного обучающего примера.

        shift:
            Сдвиг между двумя последовательными обучающими примерами.
        
        output_model="best_model.keras"
            Имя выходного файла модели.
        
        **kwargs
            Любые другие произвольные параметры, которые принимает метод fit.
    """
    (x_train, y_train), (x_test, y_test) = dataset.load_data(dataset_file, sample_size, shift)

    one_hot_encoder = layers.StringLookup()
    one_hot_encoder.adapt([y_train, y_test])
    y_train = one_hot_encoder(y_train)
    y_test = one_hot_encoder(y_test)

    callbacks = [
        ModelCheckpoint(
            filepath=output_model,
            monitor='val_loss',
            save_best_only=True
        )
    ]

    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        shuffle=False,
        callbacks=callbacks,
        **kwargs
    )

    return history


def predict(model, video_path, yolo_model, batch_size, deviation_threshold=0.1):
    """
    Выполняет прогноз по видео.

    Параметры:
        model
            Классификационная модель.

        video_path
            Путь к видео.
        
        yolo_model
            Имя pose-модели YOLO.
        
        batch_size
            Размер "пакета" кадров, на основании которого, делается прогноз модели.

        deviation_threshold=0.1
            Пороговое значение дисперсии выходного вектора, ниже которого невозможно сделать
            достоверный прогноз.
    
    Возвращает:
        Итератор предсказаний по пакетам кадров, возвращающий номер класса, если предсказание
        в достаточной степени доставерно, иначе None.
    """
    kl = KeypointsLoader(yolo_model)
    for predicion in kl(video_path, batch_size):
        predicted_class = np.std(predicion) > deviation_threshold if tf.argmax(model(predicion)[0]) else None
        yield predicted_class