from darkflow.net.build import TFNet
import cv2

# add GPU arg for testing on Jetson
options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights"}

tfnet = TFNet(options)

def label_parser(predictions):
    '''Takes in a list of dictionaries corresponding to outputs of model,
    returns array of tuples corresponding to topleft and bottomright corners of
    bounding box for objects labeled as 'person'.'''
    pass

def predict(image):
    return tfnet.return_predict(image)
