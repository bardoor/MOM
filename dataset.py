import csv
from pathlib import Path
import time

import pandas as pd
import numpy as np
from numpy import ndarray

from keypoints import KeypointsLoader
from utils import find_videos, find_missing_folders, format_time


__all__ = ('load_data', 'create_from_videos')


class _CsvFromVideos:

    def __init__(self,
                 yolo_model: str,
                 verbose: bool = True
                 ) -> None:
        self.kl = KeypointsLoader(yolo_model)
        self.verbose = verbose

    def _log_on_find_videos_end(self) -> None:
        if self.verbose:
            print(f"Found {self._total_videos_count} videos\n")

    def _log_on_open_csv_end(self,
                             csv_file: str | Path,
                             mode: str
                             ) -> None:
        if self.verbose:
            if mode == "a":
                print(f"Appending data to dataset file \"{str(csv_file)}\"\n")
            elif mode == "w":
                print(f"Created dataset file \"{str(csv_file)}\"\n")

    def _find_videos(self,
                     classes_dir: str | Path) -> None:
        self._videos = find_videos(classes_dir)
        self._total_videos_count = len(self._videos)
        self._log_on_find_videos_end()

    def _open_csv(self,
                  file_name: str | Path,
                  mode: str
                  ) -> None:
        self._dataset_file = open(file_name, mode, newline="")
        self._csv_writer = csv.writer(self._dataset_file)
        self._log_on_open_csv_end(file_name, mode)

    def _close_csv(self) -> None:
        self._dataset_file.close()

    def _log_on_process_video_end(self,
                                  video_path: str | Path,
                                  processing_time: float
                                  ) -> None:
        if self.verbose:
            formatted_time = format_time(processing_time)
            print(f"{self._processed_videos_count}/{self._total_videos_count}: processed \"{str(video_path)}\" in {formatted_time}\n")

    def _process_video(self,
                       activity: str,
                       video: str | Path
                       ) -> float:
        start_time = time.time()
        for persons_keypoints in self.kl(video):
            first_person_keypoints = persons_keypoints[0]
            self._csv_writer.writerow(
                [activity, *list(first_person_keypoints)])
        processing_time = time.time() - start_time

        self._processed_videos_count += 1

        self._log_on_process_video_end(video, processing_time)

        return processing_time

    def _log_on_process_videos_end(self) -> None:
        if self.verbose:
            formatted_time = format_time(self._total_processing_time)
            ending = "s" if self._total_videos_count > 0 else ""
            print(
                f"It took {formatted_time} to process {self._total_videos_count} video{ending}\n")

    def _log_remaining_time(self) -> None:
        if self.verbose:
            mean_processing_time = self._total_processing_time / self._processed_videos_count
            time_left = mean_processing_time * \
                (self._total_videos_count - self._processed_videos_count)
            print(f"Already processed {self._processed_videos_count} videos in {format_time(self._total_processing_time)}, " +
                  f"{self._total_videos_count - self._processed_videos_count} left...")
            print(f"I predict that there are {format_time(time_left)} left\n")

    def _process_videos(self) -> None:
        self._processed_videos_count = 0

        self._total_processing_time = 0

        for i, video in enumerate(self._videos, start=1):
            activity = video.parts[-2]
            self._total_processing_time += self._process_video(activity, video)

            if i % 5 == 0:
                self._log_remaining_time()

        self._log_on_process_videos_end()

    def __call__(self,
                 file_name: str | Path,
                 classes_dir: str | Path,
                 mode: str
                 ) -> None:
        self._open_csv(file_name, mode)
        self._find_videos(classes_dir)
        self._process_videos()
        self._close_csv()


def create_from_videos(
        output_file_name: str | Path,
        videos_dir: str | Path,
        classes: list[str],
        mode: str = "w",
        yolo_model: str = "yolov8n-pose.pt",
        verbose: bool = True
) -> None:
    """
        Создает csv-файл датасета из папок с видео.

    Параметры:
    ---------
    output_file_name: str | Path
        Имя выходного csv-файла с датасетом или путь к нему.

    videos_dir: str | Path
        Путь к папкам с видео.

    classes: list[str]
        Список папок (классов) для записи.

    mode: str = "w"
        Режим открытия файла с датасетом.

    yolo_model: str = "yolovn8-pose.pt"
        Имя pose-модель YOLO для выделения кейпоинтов из видео.

    verbose: bool = True
        Флаг, указывающий, выводить ли подробности об обработке видео.
     """
    missing_classes = find_missing_folders(videos_dir, classes)
    if missing_classes:
        raise RuntimeError(f"Couldn't found such classes: {missing_classes}")

    videos_to_csv = _CsvFromVideos(yolo_model, verbose)
    videos_to_csv(output_file_name, videos_dir, mode)


def load_data(
        csv_file: str | Path,
        size: int = 20,
        shift: int = 5,
        test_split: float = 0.2
) -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
    """
    Загружает обучающую выборку и тестовую выборку из файла датасета.

    Параметры:
    ---------
    csv_file: str | Path
        Имя csv-файла с датасетом или путь к нему.

    size: int = 20
        Размер одного обучающего примера.

    shift: int = 5
        Сдвиг между двумя последовательными обучающими примерами.
        Например, пусть видео содержит 4 кадра (рассмотрим список кадров [1, 2, 3, 4]),
        тогда при size = 2 и shift = 1 будут сгенерированы следующие примеры: [1, 2], [2, 3], [3, 4].

    test_split: float = 0.2
        Часть всей выборки, которая будет отнесена к тестовой.

    Возвращает:
    ----------
        Обучающая и тестовая выборки
    """
    if not Path(csv_file).exists():
        raise RuntimeError(f"Dataset file \"{csv_file}\" doesn't exist")

    if test_split <= 0 or test_split >= 1:
        raise RuntimeError(f"test_split should be in range (0; 1), got \"{test_split}\"")

    dataset = pd.read_csv(csv_file, header=None)

    features = dataset.iloc[:, 1:].to_numpy(dtype="f4")
    labels = dataset.iloc[:, 0].to_numpy(dtype="U")

    x_train, y_train = [], []

    for i in range(0, len(features) - size, shift):
        x = features[i:(i + size)]
        y = labels[i:(i + size)]

        # Т.к. кейпоинты в csv-датасете идут подряд, то возможна ситуация,
        # когда в результате сдвига образуется последовательность из смешанных примеров, т.е.
        # "jumps", 1.23, 4.56, ...
        # "jumps", 1.23, 4.56, ...
        # "sit-ups", 0.45, 1.23, ...
        # В случае когда размер примера равен 3 (size = 3), будет создан пример из нескольких меток,
        # что является недопустимым. Поэтому, прежде чем генерировать новый пример,
        # необходимо проверить, что были извлечены только уникальные метки.
        if y[0] != y[-1]:
            continue

        x_train.append(x)
        y_train.append(y[0])

    border = int(len(x_train) * test_split)

    x_test = np.stack(x_train[:border])
    y_test = np.stack(y_train[:border])
    x_train = np.stack(x_train[border:])
    y_train = np.stack(y_train[border:])

    return (x_train, y_train), (x_test, y_test)
