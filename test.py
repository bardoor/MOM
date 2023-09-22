from collections import Counter
import glob

from tensorflow import keras
import numpy as np

from classifier import create_model
from keypoints import KeypointsLoader


CLASSES = [
    "jumps_2",
    "jumps_left",
    "jumps_right",
    "sit-ups_narrow",
    "sit-ups_ord",
    "tilts_body",
    "tilts_head"
]


class VideoClassifier:

    def __init__(self):
        self.keypoints_loader = KeypointsLoader(yolo_model="yolov8n-pose.pt")
        self.classifier_net = create_model(classes_count=7)
        self.classifier_net.load_weights("pretrained.h5")
        self.fps = 20
        self.deviation_threshold = 0.1

    def classify(self, videos):
        results = []

        if isinstance(videos, str):
            videos = [videos]

        for video in videos:
            print(f"Processing video \"{video}\"", end="", flush=True)

            pred_classes = []

            for frames in self.keypoints_loader(video, batch_size=self.fps):
                print(".", end="", flush=True)
                pred = self.classifier_net(np.array([frames]))[0]
                pred_class = np.argmax(pred) if np.std(pred) > self.deviation_threshold else None
                pred_classes.append(pred_class)
            
            stats = Counter(pred_classes)
            most_common_class = stats.most_common(1)[0][0]

            results.append(most_common_class)

            print()

        return results
    
    def calculate_accuracy(self, label, videos):
        predicted_labels = self.classify(videos)
        return predicted_labels.count(label) / len(predicted_labels)
    

vc = VideoClassifier()


for class_ in CLASSES[1:]:
    print(f"Calculating precision for {class_}:")
    print(vc.calculate_accuracy(class_, glob.glob(f"videos/{class_}/*")))
    print()