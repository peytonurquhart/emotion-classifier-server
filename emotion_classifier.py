# Programmer: Peyton Urquhart
# Project: Image Emotion Classifier
# Date: 3/31/2022

#   emotion_classifier.py
#
#   This file can be used as a factory to create an EmotionClassifier app
#   that could be used in other programss

from predict_image import PredictImage

class EmotionClassifier:
    def __init__(self, model):
        self.clf = model
        self.predictor = PredictImage(self.clf)

    def predict_from_path(self, path):
        face_predictions = []
        preds = self.predictor.process_predict(path, True)
        for face in preds:
            face_predictions.append(self.predictor.format_probabilities(face))
        return face_predictions

    def predict_from_data(self, data):
        face_predictions = []
        preds = self.predictor.process_predict(data, False)
        for face in preds:
            face_predictions.append(self.predictor.format_probabilities(face))
        return face_predictions
