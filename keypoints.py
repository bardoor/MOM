from pathlib import Path
from typing import Iterator

from torch import cuda
import cv2
from ultralytics import YOLO
from numpy import ndarray


class KeypointsLoader:

    def __init__(self, yolo_model):
        """
        Параметры:
        ---------
        yolo_model: str
            Имя pose-модели yolo.
            В случае отсутствия локальной копии, модель будет скачана из интернета.
        """
        self.yolo_model = YOLO(yolo_model)

        if cuda.is_available():
            self.yolo_model.to("cuda")

    def __call__(self, video_path: str | Path, batch_size: int = 1) -> Iterator[ndarray]:
        """
        Получить кейпоинты со всех кадров видео, содержащих людей

        Параметры:
        ---------
        video_path: str
            Путь к видео-файлу
        
        Возвращает:
        ----------
        Итератор кейпоинтов со всех кадров видео, содержащих людей
        """
        cap = cv2.VideoCapture(str(Path(video_path)))
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                # Извлекаем с кадра кейпоинты людей (класс "0" обозначет людей)
                results = self.yolo_model(frame, classes=0, verbose=False)
                keypoints = results[0].keypoints.xyn.numpy()

                # Возвращаем кейпоинты (матрицу формы [люди, кейпоинты]), только если на кадре были люди
                if keypoints.size > 0:
                    keypoints = keypoints.reshape(-1, 34)
                    yield keypoints
            else:
                break

        cap.release()
        cv2.destroyAllWindows()