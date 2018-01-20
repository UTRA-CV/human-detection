# Human Detection

For human detection for autononomous rover.

## Getting Started

Follow the instructions below to prepare your machine.

## Prerequisites

You will need:

```
Python 3.5.x
Tensorflow
OpenCV
```

## Installation

### Python 3.5.x

Go to the Python website: [click me :D](https://www.python.org/downloads/release/python-350/)

and choose and download the version that you need.

Make sure you download Python 3.5.x for consistency on the team.

### TensorFlow

Go to the TensorFlow website: [click me :P](https://www.tensorflow.org/install/)

Select your OS and select the method that you want to install TensorFlow with.

Make sure you install TensorFlow that is compatible with Python 3.5.x.

### OpenCV

Install this package using pip3:
```
pip3 install opencv-python
```

Import the package in python shell:
```
import cv2
```

Also use pip3 to install matplotlib:
```
pip3 install matplotlib
```

If you have anymore questions go to this website: [click me :3](https://pypi.python.org/pypi/opencv-python)


## Running the tests

You can run the following tests to confirm that the programs are correctly installed.

### Python 3.5.x

Test Python by testing if you can open it on terminal

### TensorFlow

Run this program to make sure you can import TensorFlow

```
# Python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```
[source and description of the program](https://www.tensorflow.org/install/install_mac)

### OpenCV

Run this program for edge detection of an image

```
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('image.jpg',0) #make sure the image you're processing is in the same directory as your program
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
```
[source and description of the program](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html)
