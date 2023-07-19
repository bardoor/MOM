from torch import cuda
import cv2
from ultralytics import YOLO
from numpy import ndarray


class VideoKeypointsLoader:

    def __init__(self, yolo_model_name: str = "yolov8n-pose.pt"):
        """
        Параметры:
        ---------
        yolo_model_name: str
            Имя pose-модели yolo (по умолчанию - "yolov8n-pose.pt").
            В случае отсутствия локальной копии, модель будет скачана из интернета
        """

        self.yolo_model = YOLO(yolo_model_name)

        if cuda.is_available():
            self.yolo_model.to("cuda")

    def __call__(self, path: str) -> list[ndarray]:
        """
        Получить кейпоинты со всех кадров видео, содержащих людей

        Параметры:
        ---------
        path: str
            Путь к видео-файлу
        s
        Возвращает:
        ----------
        Список кейпоинтов со всех кадров видео, содержащих людей
        """

        video_keypoints = []

        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                # Извлекаем с кадра кейпоинты людей (класс "0" обозначет людей)
                results = self.yolo_model(frame, classes=0, verbose=False)
                keypoints = results[0].keypoints.xyn.numpy()

                # Помещаем кейпоинты в список, только если на кадре были люди
                if keypoints.size > 0:
                    keypoints = keypoints.reshape(-1, 34)
                    video_keypoints.append(keypoints)
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

        return video_keypoints
