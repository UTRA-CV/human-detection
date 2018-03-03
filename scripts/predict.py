from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "gpu": 1.0, "threshold": 0.5}

tfnet = TFNet(options)

def label_parser(predictions):
    '''Takes in a list of dictionaries corresponding to outputs of model,
    filters for objects labelled as 'person'.'''
    new_predictions = []
    for prediction in predictions:
        if prediction['label'] == "person":
            new_predictions.append(prediction)
    return new_predictions

def predict(image):
    return label_parser(tfnet.return_predict(image))
