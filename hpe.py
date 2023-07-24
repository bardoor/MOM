import argparse
import sys
from pprint import pprint as pp
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

import dataset
import model
from utils import show_pie_chart, show_model_accuracy


parser = argparse.ArgumentParser(description="Генерация датасета, тренировка и предсказание")
parser.add_argument("-g", "--generate", type=str, default=None,
                    help="Путь к директории, содержащей папки с видео")
parser.add_argument("-t", "--train", type=str, default=None,
                    help="Путь к csv-файлу, содержащему датасет")
parser.add_argument("-w", "--weights", type=str, default=None,
                    help="Начальные веса модели")
parser.add_argument("-p", "--predict", type=str,
                    help="Предсказание модели по видео")
parser.add_argument("-y", "--yolo", type=str, default=None,
                    help="Имя pose-модели YOLO или путь к ней")

args = parser.parse_args()

if args.generate is not None:
    if args.yolo is not None:
        dataset.create_from_videos("dataset.csv", videos_dir=args.generate)
    else:
        dataset.create_from_videos("dataset.csv", videos_dir=args.generate, yolo_model=args.yolo)

elif args.predict is not None:
    if args.weights is None:
        print("Не указаны веса модели")
        sys.exit(1)

    aem = model.ActionEstimationModel(weights=args.weights)
    show_pie_chart(aem.predict(args.predict))

elif args.train is not None:
    aem = model.ActionEstimationModel(weights=args.weights, train_dataset=args.train)
    show_model_accuracy(aem.history)
