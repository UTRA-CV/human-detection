## Intro

This is a README modified from the original [darkflow](https://github.com/thtrieu/darkflow) repo for our own purposes.

## Dependencies

Python3, tensorflow 1.0, numpy, opencv 3.
See [installation](https://github.com/UTRA-CV/human-detection/installation.md) for instructions on setting these up.

### Getting started

First, navigate to where you've cloned the repo. Then, do _one_ of the following to get started with darkflow. Note that what we are doing here is installing darkflow as a _package_.

1. Just build the Cython extensions in place. NOTE: If installing this way you will have to use `./flow` in the cloned darkflow directory instead of `flow` as darkflow is not installed globally.
    ```
    python3 setup.py build_ext --inplace
    ```

2. Let pip install darkflow globally in dev mode (still globally accessible, but changes to the code immediately take effect)
    ```
    pip install -e .
    ```

3. Install with pip globally
    ```
    pip install .
    ```

Download [weight files](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU) (all you need is `yolo.weights`) and save in the `bin/` folder.

For our purposes, we are only interested in labeling persons. We have two options: we can either train a new net with only one label 'person', or simply ignore anything not labelled as person. The only benefit of training a new net with only 'person' would be speed, which is unnecessary since prediction is already pretty fast. So, we will simply parse the output of the model for objects labelled 'person'.

## Flowing the graph using `flow`

In order to do prediction, we provide arguments to the command `flow`. Some important arguments are described below.

```bash
# Have a look options for arguments
flow --h
```

First, let's take a closer look at one of a very useful option `--load`

```bash
# Load tiny-yolo.weights
flow --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights
```

All input images from default folder `sample_img/` are flowed through the net and predictions are put in `sample_img/out/`. We can always specify more parameters for such forward passes, such as detection threshold, batch size, images folder, etc.

```bash
# Forward all images in sample_img/ using tiny yolo and 100% GPU usage
flow --imgdir sample_img/ --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights --gpu 1.0
```
json output can be generated with descriptions of the pixel location of each bounding box and the pixel location. Each prediction is stored in the `sample_img/out` folder by default. An example json array is shown below.
```bash
# Forward all images in sample_img/ using tiny yolo and JSON output.
flow --imgdir sample_img/ --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights --json
```
JSON output:
```json
[{"label":"person", "confidence": 0.56, "topleft": {"x": 184, "y": 101}, "bottomright": {"x": 274, "y": 382}},
{"label": "dog", "confidence": 0.32, "topleft": {"x": 71, "y": 263}, "bottomright": {"x": 193, "y": 353}},
{"label": "horse", "confidence": 0.76, "topleft": {"x": 412, "y": 109}, "bottomright": {"x": 592,"y": 337}}]
```
 - label: self explanatory
 - confidence: somewhere between 0 and 1 (how confident yolo is about that detection)
 - topleft: pixel coordinate of top left corner of box.
 - bottomright: pixel coordinate of bottom right corner of box.

## Using darkflow from another python application

We will utilize this functionality for the rover.

Please note that `return_predict(img)` must take an `numpy.ndarray`. Your image must be loaded beforehand and passed to `return_predict(img)`. Passing the file path won't work.

Result from `return_predict(img)` will be a list of dictionaries representing each detected object's values in the same format as the JSON output listed above.

```python
from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)

imgcv = cv2.imread("./sample_img/sample_dog.jpg")
result = tfnet.return_predict(imgcv)
print(result)
```
