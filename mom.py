import argparse
from collections import Counter
import json
import sys

import keras

import dataset
import classifier
from utils import show_pie_chart, show_model_accuracy


with open("config.json", "r") as f:
    config = json.load(f)

parser = argparse.ArgumentParser(description="Генерация датасета, тренировка и предсказание")
parser.add_argument("-d", "--dataset", type=str, default=None,
                    help="Имя csv-файла датасета")
parser.add_argument("-g", "--generate", type=str, default=None,
                    help="Путь к папкам с видео")
parser.add_argument("-t", "--train", type=str, default=None,
                    help="Путь к csv-файлу, содержащему датасет")
parser.add_argument("-m", "--model", type=str, default=None,
                    help="Путь к модели классификатора")
parser.add_argument("-p", "--predict", type=str, default=None,
                    help="Путь к видео, для которого небходимо сделать прогноз")
parser.add_argument("-v", "--verbose", type=bool, default=True,
                    help="Флаг, указывающий, выводить ли подробности об обработке данных")
parser.add_argument("-a", "--accuracy", type=bool, default=True,
                    help="Флаг, указывающий, выводить ли график кривой точности модели во время обучения")
parser.add_argument("-c", "--chart", type=bool, default=False,
                    help="Флаг, указывающий, выводить ли в конце предсказания график, вместо названия наиболее часто встречающегося класса в консоль")

args = parser.parse_args()

if args.generate is not None:
    if args.yolo is None:
        print("Не указана pose-модель YOLO")
        sys.exit(1)

    if args.dataset is None:
        print("Не указано имя csv-файла датасета")
        sys.exit(1)

    dataset.create_from_videos(
        output_file_name=args.dataset,
        videos_dir=args.generate,
        classes=config["classes"],
        yolo_model=config["yolo"],
        verbose=config["verbose"]
    )
elif args.train is not None:
    if args.dataset is None:
        print("Не указано имя csv-файла датасета")
        sys.exit(1)

    train_config = config["training"]
    classes_count = len(config["classes"])

    m = classifier.create_model(classes_count)
    history = classifier.train_model(
        model=m,
        dataset_file=args.dataset,
        sample_size=train_config["sample_size"],
        shift=train_config["shift"],
        epochs=train_config["epochs"],
        batch_size=train_config["batch_size"]
    )

    if args.accuracy:
        show_model_accuracy(history)
elif args.predict is not None:
    if args.model is None:
        print("Не указана модель классификатора")
        sys.exit(1)

    classes = config["classes"]
    default_class = config["default_class"]

    m = keras.models.load_model(args.model)
    stats = Counter(
        classes[index] if index is not None else default_class
        for index in
            classifier.predict(
                model=m,
                video_path=args.predict,
                yolo_model=config["yolo"],
                batch_size=config["sample_size"]
            )
    )

    if args.chart:
        show_pie_chart(stats.elements())
    else:
        most_common = stats.most_common(1)
        print(most_common)