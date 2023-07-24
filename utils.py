import os
import random
from pathlib import Path

import cv2
from ultralytics.data.utils import VID_FORMATS
import matplotlib.pyplot as plt


def decrease_fps(input_video_path: str, target_fps: int) -> None:
    """
    Уменьшает количество кадров в секунду в видео путём растягивания

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


def find_videos(path: str | Path, shuffle: bool = True) -> list[Path]:
    """
    Поиск видео внутри папок в указанной директории

    Параметры:
    ---------
    path: str | Path
        Путь к видео с папками, содержащими видео

    Возвращает:
    ----------
        Список всех найденных видеофайлов
    """
    videos = []

    for file in Path(path).iterdir():
        if not file.is_dir():
            continue

        for subfile in Path(file).iterdir():
            if not subfile.is_file():
                continue

            ext = subfile.suffix.lower()[1:]
            if ext in VID_FORMATS:
                videos.append(subfile)

    if shuffle:
        random.shuffle(videos)

    return videos


def find_missing_folders(directory: str | Path, folders_names: list[str]) -> list[str]:
    """
    Ищет недостающие папки в заданной директории

    Параметры:
    ---------
    directory: str | Path
        Путь к директории

    folders_names: list[str]
        Список искомых имен папок

    Возвращает:
    ----------
        Список недостатющих имен папок
    """
    found_folders = []

    for file in Path(directory).iterdir():
        if not file.is_dir():
            continue

        if file.name in folders_names:
            found_folders.append(file.name)
    
    return list(set(folders_names) - set(found_folders))


def format_time(seconds: float) -> str:
    """
    Создает текстовое представление времени из заданного количества секунд

    Параметры:
    ---------
    seconds: float
        Время в секундах
    
    Возвращает:
        Текстовое представление времени в формате "x hours y min z sec"

    Пример:
        При seconds = 71 получаем строку "1 min 11 sec"
    """
    seconds = round(seconds)

    minutes = seconds // 60
    seconds -= minutes * 60
    hours = minutes // 60
    minutes -= hours * 60

    hours_msg = f"{hours} hours " if hours > 0 else ""
    minutes_msg = f"{minutes} min " if minutes > 0 else ""
    seconds_msg = f"{seconds} sec"

    return hours_msg + minutes_msg + seconds_msg


def show_pie_chart(data):
    counts = {}
    for item in data:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1

    labels = list(counts.keys())
    values = list(counts.values())

    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.axis('equal') 
    plt.show()


def show_model_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()