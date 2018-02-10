from darkflow.net.build import TFNet
import cv2
import glob
import time

if __name__ == '__main__':
    options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

    tfnet = TFNet(options)

    filenames = [img for img in glob.glob("./imgs/*.jpg")]
    images = [cv2.imread(img) for img in filenames]


    print('Filtering the persons . . . ')
    start_time = time.time()
    for i in range(len(images)):
        results = tfnet.return_predict(images[i])  # results is in JSON format
        # filtered_results = []
        filtered_results = [item for item in results if
                            (item['label'] == 'person' and
                            item['confidence'] >= 0.2)]

        for item_found in filtered_results:
            cv2.rectangle(images[i], (item_found['topleft']['x'], item_found['topleft']['y']),
                          (item_found['bottomright']['x'], item_found['bottomright']['y']),
                          (0,255,0), 4)
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time}s')

    # Now, filtered_results only contains entries for items labelled "person"
    # for person in filtered_results:
    print('Writing images to /imgs/out/ . . . ')
    start_time = time.time()
    for i in range(len(images)):
        cv2.imwrite(f"./imgs/out/image_out_{i}.jpg", images[i])
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time}s')
