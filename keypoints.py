from pathlib import Path

import numpy as np
from torch import cuda
import cv2
from ultralytics import YOLO


class KeypointsLoader:

    def __init__(self, yolo_model):
        """
        Параметры:
        ---------
        yolo_model
            Имя pose-модели yolo.
            В случае отсутствия локальной копии, модель будет скачана из интернета.
        """
        self.yolo_model = YOLO(yolo_model)

        if cuda.is_available():
            self.yolo_model.to("cuda")

    def __call__(self, video_path, batch_size=1):
        """
        Получить кейпоинты со всех кадров видео, содержащих людей

        Параметры:
        ---------
        video_path
            Путь к видео-файлу
        
        Возвращает:
        ----------
        Итератор кейпоинтов со всех кадров видео, содержащих людей
        """
        batch = []
        cap = cv2.VideoCapture(str(Path(video_path)))
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            # Извлекаем с кадра кейпоинты людей (класс "0" обозначет людей)
            results = self.yolo_model(frame, classes=0, verbose=False)
            keypoints = results[0].keypoints.xyn.numpy()[0, :, :] # тут мы извлекаем только первого человека на кадре

            if keypoints.size > 0:
                keypoints = keypoints.reshape(34)
                batch.append(keypoints)
                
                if len(batch) == batch_size:
                    yield np.array(batch)
                    batch = []
        
        if len(batch) > 0:
            padded_frames_count = 20 - len(batch)
            for _ in range(padded_frames_count):
                batch.append(np.zeros((34,)))
            yield np.array(batch)

        cap.release()
        cv2.destroyAllWindows()