import csv as csv
import cv2
import os
from random import shuffle
import json
import time

import numpy as np
import pandas as pd
from tensorflow import convert_to_tensor, expand_dims

from keypoints import VideoKeypointsLoader

with open("config.json", "r") as config:
    data = json.load(config)

TIME_STEPS = data['yolo_handling']['TIME_STEPS']
STEP = data['yolo_handling']['STEP']
ACTIVITY_CLASSES_NUMBER = len(data['yolo_handling']['ACTIVITY_LABELS'])
ACTIVITY_LABELS = data["yolo_handling"]["ACTIVITY_LABELS"]
ACTIVITY_LABELS_MAPPING = data['yolo_handling']['ACTIVITY_LABELS_MAPPING']
ALLOWED_VIDEO_FORMATS = data['video_handling']['ALLOWED_VIDEO_FORMATS']
KEYPOINTS = data['yolo_handling']['Keypoints']

for label in ACTIVITY_LABELS_MAPPING:
    ACTIVITY_LABELS_MAPPING[label] = convert_to_tensor(
        ACTIVITY_LABELS_MAPPING[label])


def split_path_tierwise(video_file):
    """
    Разделить абсолютный путь к файлу по уровням

    Параметры:
    ---------
    file: str
        Файл...

    Возвращает:
    ----------
    Список директорий на пути к файлу
    """
    hierarchy = []
    file_path = video_file.path

    while file_path != "":
        file_path, folder = os.path.split(file_path)
        if folder != "":
            hierarchy.insert(0, folder)

    return hierarchy


def decrease_fps(input_video_path: str, target_fps: int):
    """
    Уменьшить количество кадров в секунду в видео путём растягивания

    Параметры:
    ---------
    input_video_path: str
        Путь к видео
    target_fps: int
        Необходимое количество кадров в секунду
    """
    temp_output_path = "temp_output.mp4"

    # Открываем видеофайл
    video = cv2.VideoCapture(input_video_path)

    # Считываем частоту кадров
    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= target_fps:
        return

    # На обдумывание потомкам
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc,
                          target_fps, (int(video.get(3)), int(video.get(4))))

    # Читаем и конвертируем каждый кадр видео
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        out.write(frame)

        # Пропускаем кадры чтобы достичь целевого fps
        skip_frames = round(fps / target_fps) - 1
        for _ in range(skip_frames):
            video.read()

    # Закрываем видеофайлы
    video.release()
    out.release()

    # Удаляем исходный файл
    os.remove(input_video_path)

    # Переименовываем временный файл в исходное имя файла
    os.rename(temp_output_path, input_video_path)


def _find_videos(path: str, allowed_video_formats: list[str] = ALLOWED_VIDEO_FORMATS) -> list[str]:
    """
    Поиск видео заданных форматов внутри папок в указанной директории

    Параметры:
    ---------
    path: str
        Путь к видео с папками, содержащими видео
    allowed_video_formats: list[str]
        Допустимые форматы видео

    Возвращает:
    ----------
        Список всех найденных видеофайлов
    """
    videos = []

    for directory in os.scandir(path):
        if not directory.is_dir():
            continue

        for video_file in os.scandir(directory.path):
            if not video_file.is_file():
                continue

            _, ext = os.path.splitext(video_file.path)

            if ext.lower() not in allowed_video_formats:
                continue

            # Более универсальная штука чем просто возвращать абсолютный путь
            videos.append(video_file)

    return videos


def validate_folder_names(directory: str, allowed_names: list[str]):
    '''
    Строгая проверка разрешенности имеён папок в указаной директории

    Параметры:
    ---------
    directory: str
        Путь к директории, в которой валидируются имена папок
    allowed_names: list[str]
        Список допустимых имён папок

    Возвращает:
    ----------
    invalid_folders
        Список неразрешенных папок
    '''
    invalid_folders = set()
    for entry in os.scandir(directory):
        if not entry.is_dir():
            continue

        if entry.name not in allowed_names:
            # возможно надо брать путь, но с другой стороны и так понятно где они лежат....
            invalid_folders.add(entry.name)
    return list(invalid_folders)


def to_csv(path_to_videos: str, output_file_name: str, verbose: bool = True):
    """Создаёт csv-файл, содержащий датасет из обработанных видео

    Аргументы:
        path_to_videos: Путь к директории, содержащей папки с видео

        output_file_name: Имя сгенерированного csv-файла

        verbose: Флаг, указывающий, выводить ли подробности об обработке видео (по умолчанию - True)
    """

    invalid_folders = validate_folder_names(path_to_videos, ACTIVITY_LABELS)
    if len(invalid_folders) > 0:
        print(
            f"Folders that do not match the name of any class were found in {path_to_videos}:")
        # вывод недопустимых папок через перенос строки
        print('\n'.join(invalid_folders))
        print("Generating csv aborted.")
        return

    if not output_file_name.endswith(".csv"):
        output_file_name += ".csv"

    videos = _find_videos(path_to_videos)
    shuffle(videos)

    if verbose:
        print(f"Found {len(videos)} videos")

    mode = "w"
    if os.path.exists(output_file_name):
        ans = input(
            f"[WARNING]: There's already a file called \"{output_file_name}\".\n" +
            "\t 1. Overwrite an existing file\n" +
            "\t 2. Add to the existing file\n" +
            "\t 3. Cancel csv generation\n" +
            "Please select the required action (default is 1): "
        )

        if len(ans) == 0 or int(ans) == 1:
            mode = "w"
        elif int(ans) == 2:
            mode = "a"
        else:
            print("Creating csv file aborted")
            return

    column_names = ["activity"] + [keypoint for keypoint in KEYPOINTS]

    with open(output_file_name, mode) as dataset_file:
        dataset_file_writer = csv.DictWriter(dataset_file, column_names)

        if mode == "w":
            dataset_file_writer.writeheader()

        proccessed_videos_count = 0
        processing_time_log = []
        for video in videos:
            video_proc_start = time.time()
            for frame_keypoints in build_keypoints(video):
                frame_keypoints = dict(
                    zip(list(KEYPOINTS.keys()), frame_keypoints))
                frame_keypoints['activity'] = split_path_tierwise(video)[-2]
                dataset_file_writer.writerow(frame_keypoints)

            proccessed_videos_count += 1
            if proccessed_videos_count % 5 == 0 and verbose:
                processing_time_log.append(time.time() - video_proc_start)
                time_left = np.mean(processing_time_log) * \
                    (len(videos) - proccessed_videos_count)
                print(f"Already proccessed {proccessed_videos_count} videos in {int(processing_time_log[-1])} seconds, {len(videos) - proccessed_videos_count} left...\n" +
                      f"I predict that there are {int(time_left // 3600)} hours and {int(time_left // 60 - 60 * (time_left // 3600))} minutes left")


def build_keypoints(video, yolo_model_name: str = "yolov8n-pose.pt", verbose: bool = True):
    """
    Построить ключевые точки человека для указанного видео

    Параметры:
    ---------
    video
        Обрабатываемое видео
    yolo_model_name: str
        Используемая модель YOLO-pose
    verbose: bool
        Флаг для вывода подробностей об обработке видео

    Возвращает:
    ----------
    Генератор ключевых точек человека для каждого кадра видео
    """
    keypoints_loader = VideoKeypointsLoader(yolo_model_name)

    video_path = os.path.abspath(video.path)

    if verbose:
        print(f"Processing \"{video.path}\"...")

    video_keypoints = keypoints_loader(video_path)

    for frame_keypoints in video_keypoints:
        first_person_keypoints = frame_keypoints[0]
        yield first_person_keypoints


def generate(path_to_video, time_steps=TIME_STEPS, step=STEP):
    """Генерирует последовательности кадров (кейпоинтов) из видео
    с заданным размером и шагом

    Аргументы:
        path_to_video: Путь к видео

        time_steps: Сколько массивов кейпоинтов будет содержаться
        в одной последовательности (по умолчанию - 20)

        step: Насколько кадров (кейпоинтов) следующая последовательность
        опережает текущую (по умолчанию - 5)
    """

    keypoints_loader = VideoKeypointsLoader("yolo8x-pose-p6.pt")

    keypoints = keypoints_loader.load(path_to_video)
    for i in range(0, len(keypoints) - time_steps, step):
        seq = expand_dims(convert_to_tensor(
            keypoints[i:(i + time_steps)]), axis=0)
        yield seq


class Dataset:

    def __init__(self, videos: str = None, classes: list[str] = None, csv_file: str = None, time_steps: int = 20, step: int = 5):
        """
        Параметры:
        ---------
        videos: str
            Путь к папкам с видео

        classes: list[str]
            Названия папок с видео, из которых нужно создать датасет

        csv_file: str
            Имя csv датасет-файла

        time_steps: int
            Количество кадров, которое содержит один пример из датасета (по умолчанию - 20)

        step: int
            Количество кадров на которое отстаёт два последовательных примера друг от друга (по умолчанию - 5).
            Например, пусть видео содержит 4 кадра (рассмотрим список кадров [1, 2, 3, 4]),
            тогда при time_steps = 2 и step = 1 будут сгенерированы следующие примеры: [1, 2], [2, 3], [3, 4]
        """
        if videos is None and csv is None:
            raise RuntimeError(
                "Dataset should be initialized with either videos or csv")

        if videos is not None and csv is not None:
            raise RuntimeError(
                "Dataset should be initialized with only videos or csv")

        if videos is not None and classes is None:
            raise RuntimeError("No classes were provided for dataset")

        if videos is None and classes is not None:
            raise RuntimeError("No videos were provided for dataset")

        if csv is not None and classes is not None:
            raise RuntimeError("Can't specify classes for csv file")

        if not isinstance(time_steps, int) or not isinstance(step, int) or time_steps <= 0 or step <= 0:
            raise RuntimeError(
                "Time steps and step should be positive integers")

        self.videos = videos
        self.classes = classes
        self.csv_file = csv_file
        self.time_steps = time_steps
        self.step = step

    def to_csv(self, output_file: str, mode: str = "w", verbose: bool = True):
        """
        Создаёт csv-файл, содержащий датасет из обработанных видео

        Параметры:
        ---------
        output_file: str
            Имя csv файла

        mode: str
            Режим открытия для csv файла, совпадает по значениям с режимами
            открытия файлов в функции open (по умолчанию - "w")

        verbose: bool
            Флаг, указывающий, выводить ли подробности об обработке видео (по умолчанию - True)
        """
        if self.videos is None:
            raise RuntimeError("No videos were provided for dataset")

        invalid_folders = validate_folder_names(self.videos, self.classes)
        if len(invalid_folders) > 0:
            raise RuntimeError("Folders that do not match the name of any class were found in \"{self.videos}\":\n" +
                               "\n".join(invalid_folders))

        if not output_file_name.endswith(".csv"):
            output_file_name += ".csv"

        videos = _find_videos(self.videos)
        shuffle(videos)

        if verbose:
            print(f"Found {len(videos)} videos")

        column_names = ["activity"] + [keypoint for keypoint in KEYPOINTS]

        with open(output_file_name, mode) as dataset_file:
            dataset_file_writer = csv.DictWriter(dataset_file, column_names)

            if mode == "w":
                dataset_file_writer.writeheader()

            proccessed_videos_count = 0
            processing_time_log = []
            for video in videos:
                video_proc_start = time.time()
                for frame_keypoints in build_keypoints(video):
                    frame_keypoints = dict(
                        zip(list(KEYPOINTS.keys()), frame_keypoints))
                    frame_keypoints['activity'] = split_path_tierwise(
                        video)[-2]
                    dataset_file_writer.writerow(frame_keypoints)

                proccessed_videos_count += 1
                if proccessed_videos_count % 5 == 0 and verbose:
                    processing_time_log.append(time.time() - video_proc_start)
                    time_left = np.mean(processing_time_log) * \
                        (len(videos) - proccessed_videos_count)
                    print(f"Already proccessed {proccessed_videos_count} videos in {int(processing_time_log[-1])} seconds, {len(videos) - proccessed_videos_count} left...\n" +
                          f"I predict that there are {int(time_left // 3600)} hours and {int(time_left // 60 - 60 * (time_left // 3600))} minutes left")

    def load_data(self):
        """
        Загружает датасет в виде матриц numpy
        """
        if self.csv_file is None:
            raise RuntimeError("No csv file was provided for dataset")

        Xs, ys = [], []

        df = pd.read_csv(self.csv_file)
        X = df.drop(columns=["activity"]).to_numpy()
        y = df.activity.to_numpy()

        for i in range(0, len(X) - self.time_steps, self.step):
            series = X[i:(i + self.time_steps)]
            labels = y[i:(i + self.time_steps)]

            # Т.к. кейпоинты в csv-датасете идут подряд, то возможна ситуация,
            # когда в результате сдвига образуется последовательность из смешанных примеров, т.е.
            # "jumps", 1.23, 4.56, ...
            # "jumps", 1.23, 4.56, ...
            # "sit-ups", 0.45, 1.23, ...
            # В случае когда размер примера равен 3 (time_steps = 3), будет создан пример из нескольких меток
            # что является недопустимым. Поэтому, прежде чем генерировать новый пример,
            # необходимо проверить, что были извлечены только уникальные метки
            if len(pd.unique(labels)) != 1:
                continue

            Xs.append(series)
            ys.append(labels[0])

        return np.array(Xs), np.array(ys)
