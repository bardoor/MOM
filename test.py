from collections import Counter

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
    
    def f(self, v):
        for frames in self.keypoints_loader(v, 20):
            print(frames.shape)

    def classify(self, videos):
        results = []

        if isinstance(videos, str):
            videos = [videos]

        for video in videos:
            print("in videos")

            pred_classes = []

            for frames in self.keypoints_loader(video, batch_size=self.fps):
                print("in frames")
                pred = self.classifier_net(np.array([frames]))[0]
                pred_class = np.std(pred) > self.deviation_threshold if np.argmax(pred) else None
                pred_classes.append(pred_class)
            
            stats = Counter(pred_classes)
            most_common_class = stats.most_common(1)[0]

            print("common")

            if most_common_class is None:
                results.append("unknown")
            else:
                results.append(CLASSES[most_common_class])


        return results
    

vc = VideoClassifier()
print(vc.classify("C:\Projects\Repos\MOM\dataset\jumps_2\document_5253781099443663757.mp4"))